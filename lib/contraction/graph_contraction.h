/******************************************************************************
 * contraction.h
 *
 * Contraction of distributed graph
 ******************************************************************************
 * Copyright (C) 2017 Sebastian Lamm <lamm@kit.edu>
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along with
 * this program.  If not, see <http://www.gnu.org/licenses/>.
 *****************************************************************************/

#ifndef _CONTRACTION_H_
#define _CONTRACTION_H_

#include <iostream>
#include <google/dense_hash_map>

#include "config.h"
#include "definitions.h"
#include "graph_access.h"
#include "edge_hash.h"

class Contraction {
 public:
  Contraction(GraphAccess &g, const PEID rank, const PEID size)
      : g_(g), rank_(rank), size_(size),
        num_smaller_components_(0),
        num_local_components_(0),
        num_global_components_(0),
        node_buffers_(size),
        edge_buffers_(size) {
    // local_edges_.set_empty_key(HashedEdge{0, 0, 0, 0});
  }
  virtual ~Contraction() = default;

  GraphAccess BuildComponentAdjacencyGraph() {
    Timer t;
    t.Restart();
    if (rank_ == ROOT) std::cout << "[STATUS] |- Compute component prefix sum (" << t.Elapsed() << ")" << std::endl;
    ComputeComponentPrefixSum();
    if (rank_ == ROOT) std::cout << "[STATUS] |- Compute local contraction mapping (" << t.Elapsed() << ")" << std::endl;
    ComputeLocalContractionMapping();
    if (rank_ == ROOT) std::cout << "[STATUS] |- Exchange ghost contraction mapping (" << t.Elapsed() << ")" << std::endl;
    ExchangeGhostContractionMapping();

    if (rank_ == ROOT) std::cout << "[STATUS] |- Generate local contraction edges (" << t.Elapsed() << ")" << std::endl;
    GenerateLocalContractionEdges();
    if (rank_ == ROOT) std::cout << "[STATUS] |- Exchange ghost contraction edges (" << t.Elapsed() << ")" << std::endl;
    ExchangeGhostContractionEdges();

    if (rank_ == ROOT) std::cout << "[STATUS] |- Build local contration graph (" << t.Elapsed() << ")" << std::endl;
    return BuildLocalContractionGraph();
  }

 private:
  // Original graph instance
  GraphAccess &g_;

  // Network information
  PEID rank_, size_;

  // Component information
  VertexID num_smaller_components_, num_local_components_,
      num_global_components_;
  std::unordered_set<VertexID> local_components_{};

  // Local edges
  EdgeHash local_edges_{};

  // Send buffers
  std::vector<std::vector<VertexID>> node_buffers_;
  std::vector<std::vector<VertexID>> edge_buffers_;

  void ComputeComponentPrefixSum() {
    // Gather local components O(max(#component))
    num_local_components_ = 0;
    g_.ForallLocalVertices([&](const VertexID v) {
      VertexID v_label = g_.GetVertexLabel(v);
      if (local_components_.find(v_label) == end(local_components_)) {
        local_components_.insert(v_label);
        num_local_components_++;
      }
    });

    // Build prefix sum over local components O(log P)
    VertexID component_prefix_sum;
    MPI_Scan(&num_local_components_,
             &component_prefix_sum,
             1,
             MPI_UNSIGNED_LONG,
             MPI_SUM,
             MPI_COMM_WORLD);

    num_global_components_ = component_prefix_sum;
    MPI_Bcast(&num_global_components_,
              1,
              MPI_UNSIGNED_LONG,
              size_ - 1,
              MPI_COMM_WORLD);

    num_smaller_components_ = component_prefix_sum - num_local_components_;
  }

  void ComputeLocalContractionMapping() {
    // Map local components to contraction vertices O(n/P)
    std::unordered_map<VertexID, VertexID> label_map;
    VertexID current_component = num_smaller_components_;
    for (const VertexID c : local_components_) {
      label_map[c] = current_component++;
    }

    // Setup contraction vertices for local vertices O(n/P)
    g_.AllocateContractionVertices();
    g_.ForallLocalVertices([&](const VertexID v) {
      g_.SetContractionVertex(v, label_map[g_.GetVertexLabel(v)]);
    });
  }

  void ExchangeGhostContractionMapping() {
    std::vector<bool> packed_pes(size_, false);
    std::vector<bool> largest_pes(size_, false);

    // Optimization: Don't send largest component between pairs of PEs
    // std::unordered_map<VertexID, VertexID> component_sizes;
    // std::unordered_map<PEID, VertexID> largest_component_sizes;
    // std::unordered_map<PEID, VertexID> largest_component_ids;
    // for (const auto &pe : g_.GetAdjacentPEs()) {
    //   largest_component_sizes[pe] = 0;
    //   largest_component_ids[pe] = 0;
    // }

    google::dense_hash_map<VertexID, VertexID> component_sizes;
    component_sizes.set_empty_key(std::numeric_limits<VertexID>::max());
    std::vector<VertexID> largest_component_sizes(size_, 0);
    std::vector<VertexID> largest_component_ids(size_, 0);

    g_.ForallLocalVertices([&](const VertexID v) {
      if (g_.IsInterface(v)) {
        VertexID cv = g_.GetContractionVertex(v);
        if (component_sizes.find(cv) == component_sizes.end()) 
          component_sizes[cv] = 0;
        component_sizes[cv]++;
        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (g_.IsGhost(w)) {
            PEID target_pe = g_.GetPE(w);
            if (component_sizes[cv] > largest_component_sizes[target_pe]) {
              largest_component_sizes[target_pe] = component_sizes[cv];
              largest_component_ids[target_pe] = cv;
            } 
          }
        });
      }
    });

    // for (const auto &pe : g_.GetAdjacentPEs()) {
    for (PEID i = 0; i < size_; ++i) {
      if (largest_component_sizes[i] > 0) {
        node_buffers_[i].push_back(std::numeric_limits<VertexID>::max()); 
        node_buffers_[i].push_back(largest_component_ids[i]); 
      }
    }

    // Collect remaining ghost vertex updates O(n/P)
    g_.ForallLocalVertices([&](const VertexID v) {
      if (g_.IsInterface(v)) {
        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (g_.IsGhost(w)) {
            PEID target_pe = g_.GetPE(w);
            if (!packed_pes[target_pe]) {
              if (g_.GetContractionVertex(v) != largest_component_ids[target_pe]) {
                node_buffers_[target_pe].push_back(g_.GetGlobalID(v));
                node_buffers_[target_pe].push_back(g_.GetContractionVertex(v));
                packed_pes[target_pe] = true;
              }
            }
          }
        });

        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (g_.IsGhost(w)) {
            PEID target_pe = g_.GetPE(w);
            packed_pes[target_pe] = false;
          }
        });
      }
    });

    // Send ghost vertex updates O(cut size) (communication)
    for (PEID i = 0; i < size_; ++i) {
      if (g_.IsAdjacentPE(i)) {
        if (node_buffers_[i].empty()) node_buffers_[i].push_back(0);
        MPI_Request req;
        MPI_Isend(&node_buffers_[i][0],
                  static_cast<int>(node_buffers_[i].size()),
                  MPI_UNSIGNED_LONG,
                  i,
                  i + 6 * size_,
                  MPI_COMM_WORLD,
                  &req);
      };
    }

    // Receive updates O(cut size)
    // Optimization for receiving largest component
    PEID num_adjacent_pes = g_.GetNumberOfAdjacentPEs();
    PEID recv_messages = 0;
    std::vector<bool> relabled(g_.GetNumberOfVertices(), false);
    std::vector<VertexID> largest_components(size_, 0);
    while (recv_messages < num_adjacent_pes) {
      MPI_Status st{};
      MPI_Probe(MPI_ANY_SOURCE, rank_ + 6 * size_, MPI_COMM_WORLD, &st);

      int message_length;
      MPI_Get_count(&st, MPI_UNSIGNED_LONG, &message_length);
      std::vector<VertexID> message(static_cast<unsigned long>(message_length));

      MPI_Status rst{};
      MPI_Recv(&message[0],
               message_length,
               MPI_UNSIGNED_LONG,
               st.MPI_SOURCE,
               rank_ + 6 * size_,
               MPI_COMM_WORLD,
               &rst);
      recv_messages++;

      for (int i = 0; i < message_length - 1; i += 2) {
        VertexID global_id = message[i];
        VertexID contraction_id = message[i + 1];
        if (global_id == std::numeric_limits<VertexID>::max())
          largest_components[st.MPI_SOURCE] = contraction_id;
        else  {
          g_.SetContractionVertex(g_.GetLocalID(global_id), contraction_id);
          relabled[g_.GetLocalID(global_id)] = true;
        }
      }
    }
    
    // Apply largest component to all unlabeled vertices
    g_.ForallGhostVertices([&](const VertexID v) {
      if (!relabled[v])
        g_.SetContractionVertex(v, largest_components[g_.GetPE(v)]);
    });
  }

  void GenerateLocalContractionEdges() {
    // Gather local edges (there shouldn't be any) O(m/P)
    g_.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = g_.GetContractionVertex(v);
      g_.ForallNeighbors(v, [&](const VertexID w) {
        VertexID cw = g_.GetContractionVertex(w);
        if (cv != cw)
          local_edges_.insert(HashedEdge{num_global_components_, cv, cw, g_.GetPE(w)});
      });
    });
  }

  void ExchangeGhostContractionEdges() {
    // Determine edge targets O(cut size)
    for (const auto &e : local_edges_) {
      edge_buffers_[e.rank].push_back(e.target);
      edge_buffers_[e.rank].push_back(e.source);
    }

    // Send edges O(cut size) (communication)
    for (PEID i = 0; i < size_; ++i) {
      if (g_.IsAdjacentPE(i)) {
        if (edge_buffers_[i].empty()) edge_buffers_[i].push_back(0);
        MPI_Request req;
        MPI_Isend(&edge_buffers_[i][0],
                  static_cast<int>(edge_buffers_[i].size()),
                  MPI_UNSIGNED_LONG,
                  i,
                  i + 7 * size_,
                  MPI_COMM_WORLD,
                  &req);
      }
    }

    // Receive updates O(cut size)
    PEID num_adjacent_pes = g_.GetNumberOfAdjacentPEs();
    PEID recv_messages = 0;
    while (recv_messages < num_adjacent_pes) {
      MPI_Status st{};
      MPI_Probe(MPI_ANY_SOURCE, rank_ + 7 * size_, MPI_COMM_WORLD, &st);

      int message_length;
      MPI_Get_count(&st, MPI_UNSIGNED_LONG, &message_length);
      std::vector<VertexID> message(static_cast<unsigned long>(message_length));

      MPI_Status rst{};
      MPI_Recv(&message[0],
               message_length,
               MPI_UNSIGNED_LONG,
               st.MPI_SOURCE,
               rank_ + 7 * size_,
               MPI_COMM_WORLD,
               &rst);
      recv_messages++;

      // std::cout << "[R" << rank_ << "] recv msg " << recv_messages << "/" << num_adjacent_pes << " from " << st.MPI_SOURCE << " with length " << message_length << std::endl;

      if (message_length == 1) continue;
      for (int i = 0; i < message_length - 1; i += 2) {
        VertexID source = message[i];
        VertexID target = message[i + 1];
        local_edges_.insert(HashedEdge{num_global_components_, source, target,
                                        st.MPI_SOURCE});
      }
    }
  }

  GraphAccess BuildLocalContractionGraph() {
    VertexID from = num_smaller_components_;
    VertexID to = num_smaller_components_ + num_local_components_ - 1;

    EdgeID edge_counter = 0;
    std::vector<std::vector<std::pair<VertexID, VertexID>>>
        local_edge_lists(num_local_components_);
    for (const auto &e : local_edges_) {
      if (from <= e.target && e.target < to) {
        local_edge_lists[e.source - from].emplace_back(e.target - from, rank_);
        local_edge_lists[e.target - from].emplace_back(e.source - from, rank_);
        edge_counter += 2;
      } else {
        local_edge_lists[e.source - from].emplace_back(e.target, e.rank);
        edge_counter++;
      }
    }

    // TODO: We need to fill range array. Alternatively send pe with edges.
    GraphAccess cg(rank_, size_);
    cg.StartConstruct(num_local_components_,
                      edge_counter,
                      num_smaller_components_);

    for (VertexID i = 0; i < num_local_components_; ++i) {
      VertexID v = cg.AddVertex();
      // cg.SetVertexLabel(v, from + v);
      cg.SetVertexPayload(v, {cg.GetVertexDeviate(v), from + v, rank_});

      for (auto &j : local_edge_lists[i]) {
        VertexID target = j.first;
        cg.AddEdge(v, target, static_cast<PEID>(j.second));
      }
    }
    cg.FinishConstruct();
    return cg;
  }
};

#endif
