/******************************************************************************
 * initial_contraction.h
 *
 * Initial contraction of distributed graph
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

#ifndef _CAG_BUILDER_H_
#define _CAG_BUILDER_H_

#include <iostream>
#include <google/sparse_hash_set>
#include <google/dense_hash_set>

#include "config.h"
#include "definitions.h"
#include "dynamic_graph_comm.h"
#include "static_graph.h"
#include "edge_hash.h"

template <typename GraphInputType>
class CAGBuilder {
 public:
  CAGBuilder(GraphInputType &g, std::vector<VertexID> &vertex_labels, const PEID rank, const PEID size)
      : g_(g), vertex_labels_(vertex_labels), rank_(rank), size_(size),
        num_smaller_components_(0),
        num_local_components_(0),
        num_global_components_(0),
        vertex_buffers_(size),
        comm_time_(0.0) {
    local_components_.set_empty_key(-1);
    vertex_buffers_.set_empty_key(-1);
    offset_ = g_.GatherNumberOfGlobalVertices();
  }
  virtual ~CAGBuilder() = default;

  DynamicGraphCommunicator BuildDynamicComponentAdjacencyGraph() {
    PerformContraction();
    return BuildDynamicContractionGraph();
  }

  StaticGraph BuildStaticComponentAdjacencyGraph() {
    PerformContraction();
    return BuildStaticContractionGraph();
  }

  float GetCommTime() {
    return comm_time_;
  }

 private:
  // Original graph instance
  GraphInputType &g_;
  std::vector<VertexID> &vertex_labels_;

  // Network information
  PEID rank_, size_;

  // Offset for pairing
  VertexID offset_;

  // Component information
  VertexID num_smaller_components_, 
           num_local_components_,
           num_global_components_;

  google::dense_hash_set<VertexID> local_components_; 

  // Local edges
  EdgeHash local_edges_{};
  std::vector<std::tuple<VertexID, VertexID, PEID>> edges_;

  // Send buffers
  google::dense_hash_map<PEID, std::vector<VertexID>> vertex_buffers_;
  std::vector<bool> received_message_;

  // Statistics
  float comm_time_;
  Timer contraction_timer_;
  Timer comm_timer_;

  void PerformContraction() {
    contraction_timer_.Restart();
    comm_timer_.Restart();
    ComputeComponentPrefixSum();
    comm_time_ += comm_timer_.Elapsed();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Computing component prefix sum took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    contraction_timer_.Restart();
    ComputeLocalContractionMapping();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Computing local contraction mapping took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }

    contraction_timer_.Restart();
    ExchangeGhostContractionMapping();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Exchanging ghost contraction mapping took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }
    
    contraction_timer_.Restart();
    GenerateLocalContractionEdges();
    if (rank_ == ROOT) {
      std::cout << "[STATUS] |-- Generating contraction edges took " 
                << "[TIME] " << contraction_timer_.Elapsed() << std::endl;
    }
  }

  void ComputeComponentPrefixSum() {
    // Gather local components O(max(#component))
    num_local_components_ = FindLocalComponents();

    // Build prefix sum over local components O(log P)
    VertexID component_prefix_sum;
    MPI_Scan(&num_local_components_,
             &component_prefix_sum,
             1,
             MPI_VERTEX,
             MPI_SUM,
             MPI_COMM_WORLD);

    // Broadcast number of global components
    num_global_components_ = component_prefix_sum;
    MPI_Bcast(&num_global_components_,
              1,
              MPI_VERTEX,
              size_ - 1,
              MPI_COMM_WORLD);

    num_smaller_components_ = component_prefix_sum - num_local_components_;
  }

  VertexID FindLocalComponents() {
    // Add local components to hash set
    g_.ForallLocalVertices([&](const VertexID v) {
      VertexID v_label = vertex_labels_[v];
      if (local_components_.find(v_label) == end(local_components_)) {
        local_components_.insert(v_label);
      }
    });
    return local_components_.size();
  }

  void ComputeLocalContractionMapping() {
    // Map local components to contraction vertices O(n/P)
    google::dense_hash_map<VertexID, VertexID> label_map; 
    label_map.set_empty_key(-1);
    VertexID current_component = num_smaller_components_;
    for (const VertexID c : local_components_) {
      label_map[c] = current_component++;
    }

    // Setup contraction vertices for local vertices O(n/P)
    g_.AllocateContractionVertices();
    g_.ForallLocalVertices([&](const VertexID v) {
      VertexID component = label_map[vertex_labels_[v]];
      g_.SetContractionVertex(v, component);
    });
  }

  void ExchangeGhostContractionMapping() {
    IdentifyLargestInterfaceComponents();
    AddComponentMessages();

    // Send ghost vertex updates O(cut size) (communication)
    comm_timer_.Restart();
    std::vector<MPI_Request> requests;
    SendBuffers(requests);

    google::dense_hash_map<PEID, VertexID> largest_component; 
    google::dense_hash_map<VertexID, VertexID> vertex_message; 
    largest_component.set_empty_key(-1); 
    vertex_message.set_empty_key(-1);

    received_message_.resize(g_.GetNumberOfVertices(), false);
    ReceiveBuffers(largest_component, vertex_message);
    comm_time_ += comm_timer_.Elapsed();

    ApplyUpdatesToGhostVertices(largest_component, vertex_message);
    comm_timer_.Restart();
    CheckRequests(requests);
    comm_time_ += comm_timer_.Elapsed();
  }

  // TODO: Do this per PE?
  void IdentifyLargestInterfaceComponents() {
    // Compute sizes of components for interface vertices
    google::dense_hash_map<VertexID, VertexID> interface_component_size;
    interface_component_size.set_empty_key(-1);

    g_.ForallLocalVertices([&](const VertexID v) {
      if (g_.IsInterface(v)) {
        VertexID cv = g_.GetContractionVertex(v);
        if (interface_component_size.find(cv) == interface_component_size.end()) 
          interface_component_size[cv] = 0;
        interface_component_size[cv]++;
      }
    });

    std::vector<VertexID> largest_component_size(size_, 0);
    std::vector<VertexID> largest_component_id(size_, 0);

    // Identify largest component for each adjacent PE
    g_.ForallLocalVertices([&](const VertexID v) {
      if (g_.IsInterface(v)) {
        VertexID cv = g_.GetContractionVertex(v);
        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (!g_.IsLocal(w)) {
            PEID target_pe = g_.GetPE(w);
            if (interface_component_size[cv] > largest_component_size[target_pe]) {
              largest_component_size[target_pe] = interface_component_size[cv];
              largest_component_id[target_pe] = cv;
            } 
          }
        });
      }
    });

    g_.ForallAdjacentPEs([&](const PEID &pe) {
      if (largest_component_size[pe] > 0) {
        vertex_buffers_[pe].push_back(std::numeric_limits<VertexID>::max() - 1); 
        vertex_buffers_[pe].push_back(largest_component_id[pe]); 
      }
    });
  }

  void AddComponentMessages() {
    // Helper functions
    auto pair = [&](VertexID x, VertexID y) {
      return x * offset_ + y;
    };

    // Gather components with the same neighbor
    // TODO: Can we do this more efficiently?
    google::dense_hash_map<PEID, google::sparse_hash_set<VertexID>> unique_neighbors;
    unique_neighbors.set_empty_key(-1);
    VertexID buffer_size = 0;
    g_.ForallLocalVertices([&](const VertexID v) {
      if (g_.IsInterface(v)) {
        google::dense_hash_set<PEID> receiving_pes;
        receiving_pes.set_empty_key(-1);
        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (!g_.IsLocal(w)) {
            PEID target_pe = g_.GetPE(w);
            VertexID contraction_vertex = g_.GetContractionVertex(v);
            VertexID largest_component = vertex_buffers_[target_pe][1];
            // Only send message if not part of largest component
            if (contraction_vertex != largest_component) {
              // Avoid duplicates by hashing the message
              VertexID comp_pair = pair(w, g_.GetContractionVertex(v));
              if (unique_neighbors[target_pe].find(comp_pair) == end(unique_neighbors[target_pe]) &&
                  receiving_pes.find(target_pe) == end(receiving_pes)) {
                unique_neighbors[target_pe].insert(comp_pair);
                receiving_pes.insert(target_pe);
                vertex_buffers_[target_pe].push_back(g_.GetGlobalID(v));
                vertex_buffers_[target_pe].push_back(contraction_vertex);
                buffer_size++;
              }
            }
          }
        });
      }
    });
    std::cout << "R" << rank_ << " buffer size " << buffer_size << std::endl;
  }

  void SendBuffers(std::vector<MPI_Request> &requests) {
    requests.clear();

    g_.ForallAdjacentPEs([&](const PEID &pe) {
      if (vertex_buffers_[pe].empty()) {
        vertex_buffers_[pe].emplace_back(std::numeric_limits<VertexID>::max());
        vertex_buffers_[pe].emplace_back(0);
      }
      requests.emplace_back(MPI_Request());
      MPI_Isend(&vertex_buffers_[pe][0],
                static_cast<int>(vertex_buffers_[pe].size()),
                MPI_VERTEX, pe, pe + 42 * size_,
                MPI_COMM_WORLD, &requests.back());
    });
  }

  void ReceiveBuffers(google::dense_hash_map<PEID, VertexID> & largest_component, 
                      google::dense_hash_map<VertexID, VertexID> & vertex_message) {
    // Optimization for receiving largest component
    // Receive updates O(cut size)
    PEID num_adjacent_pes = g_.GetNumberOfAdjacentPEs();
    PEID recv_messages = 0;
    while (recv_messages < num_adjacent_pes) {
      MPI_Status st{};
      MPI_Probe(MPI_ANY_SOURCE, rank_ + 42 * size_, MPI_COMM_WORLD, &st);

      int message_length;
      MPI_Get_count(&st, MPI_VERTEX, &message_length);
      std::vector<VertexID> message(static_cast<unsigned long>(message_length));

      MPI_Status rst{};
      MPI_Recv(&message[0],
               message_length,
               MPI_VERTEX,
               st.MPI_SOURCE,
               rank_ + 42 * size_,
               MPI_COMM_WORLD,
               &rst);
      recv_messages++;

      if (message_length < 2) continue;
      for (int i = 0; i < message_length - 1; i += 2) {
        VertexID global_id = message[i];
        VertexID contraction_id = message[i + 1];

        if (global_id == std::numeric_limits<VertexID>::max()) continue;

        if (global_id == std::numeric_limits<VertexID>::max() - 1) {
          largest_component[st.MPI_SOURCE] = contraction_id;
        } else {
          vertex_message[g_.GetLocalID(global_id)] = contraction_id;
          received_message_[g_.GetLocalID(global_id)] = true;
        }
      }
    }
  }

  void ApplyUpdatesToGhostVertices(google::dense_hash_map<PEID, VertexID> & largest_component, 
                                   google::dense_hash_map<VertexID, VertexID> & vertex_message) {
    g_.ForallGhostVertices([&](const VertexID v) {
      PEID pe = g_.GetPE(v);
      VertexID cid = vertex_message[v];
      if (received_message_[v]) {
        g_.SetContractionVertex(v, cid);
      } else {
        g_.SetContractionVertex(v, largest_component[pe]);
      }
    });

    g_.ForallLocalVertices([&](const VertexID v) {
      google::dense_hash_map<PEID, VertexID> component_id;
      component_id.set_empty_key(-1);
      if (g_.IsInterface(v)) {
        // Store largest component for each PE
        for (auto &kv : largest_component) {
          component_id[kv.first] = kv.second;
        }
        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (g_.IsGhost(w)) {
            PEID pe = g_.GetPE(w);
            VertexID cid = g_.GetContractionVertex(w);
            // There are two adjacent vertices on the same PE with different component IDs?
            if (cid != component_id[pe] && cid != largest_component[pe]) {
              component_id[pe] = cid;
            }
          }
        });
        g_.ForallNeighbors(v, [&](const VertexID w) {
          if (g_.IsGhost(w)) {
            PEID pe = g_.GetPE(w);
            VertexID cid = g_.GetContractionVertex(w);
            if (cid != component_id[pe]) {
              g_.SetContractionVertex(w, component_id[pe]);
            }
          }
        });
      }
    });
  }

  void CheckRequests(std::vector<MPI_Request> &requests) {
    for (unsigned int i = 0; i < requests.size(); ++i) {
      if (requests[i] != MPI_REQUEST_NULL) {
        MPI_Request_free(&requests[i]);
      }
    }
  }

  void GenerateLocalContractionEdges() {
    // Gather local edges (there shouldn't be any) O(m/P)
    g_.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = g_.GetContractionVertex(v);
      g_.ForallNeighbors(v, [&](const VertexID w) {
        VertexID cw = g_.GetContractionVertex(w);
        if (cv != cw) {
          PEID pe = g_.GetPE(w);
          auto h_edge = HashedEdge{num_global_components_, cv, cw, pe};
          if (local_edges_.find(h_edge) == end(local_edges_)) {
            local_edges_.insert(h_edge);
            edges_.emplace_back(cv, cw, pe);
            edges_.emplace_back(cw, cv, rank_);
          }
        }
      });
    });
  }

  DynamicGraphCommunicator BuildDynamicContractionGraph() {
    VertexID from = num_smaller_components_;
    VertexID to = num_smaller_components_ + num_local_components_ - 1;

    google::dense_hash_map<VertexID, PEID> ghost_pe; 
    ghost_pe.set_empty_key(-1);

    // Identify ghost vertices
    for (auto &e : edges_) {
      VertexID target = std::get<1>(e);
      PEID pe = std::get<2>(e);
      if (std::get<2>(e) != rank_) {
        if (ghost_pe.find(target) == ghost_pe.end()) {
            ghost_pe[target] = pe;
        } 
      }
    }

    VertexID number_of_ghost_vertices = ghost_pe.size();
    VertexID number_of_edges = edges_.size();

    DynamicGraphCommunicator cg(rank_, size_);
    cg.StartConstruct(num_local_components_,
                      number_of_ghost_vertices,
                      num_global_components_);

    // Initialize local vertices
    for (VertexID v = 0; v < num_local_components_; v++) {
        cg.AddVertex(from + v);
        cg.SetVertexLabel(v, from + v);
        cg.SetVertexRoot(v, rank_);
    }

    // Initialize ghost vertices
    for (const auto &kv : ghost_pe) {
      cg.AddGhostVertex(kv.first, kv.second);
    }

    for (auto &edge : edges_) {
      cg.AddEdge(cg.GetLocalID(std::get<0>(edge)), std::get<1>(edge), std::get<2>(edge));
    }

    cg.FinishConstruct();
    return cg;
  }

  StaticGraph BuildStaticContractionGraph() {
    VertexID from = num_smaller_components_;
    VertexID to = num_smaller_components_ + num_local_components_ - 1;

    google::dense_hash_map<VertexID, PEID> ghost_pe; 
    ghost_pe.set_empty_key(-1);

    // Identify ghost vertices
    for (auto &e : edges_) {
      VertexID target = std::get<1>(e);
      PEID pe = std::get<2>(e);
      if (std::get<2>(e) != rank_) {
        if (ghost_pe.find(target) == ghost_pe.end()) {
            ghost_pe[target] = pe;
        } 
      }
    }

    VertexID number_of_ghost_vertices = ghost_pe.size();
    VertexID number_of_edges = edges_.size();

    StaticGraph cg(rank_, size_);
    cg.StartConstruct(num_local_components_,
                      number_of_ghost_vertices,
                      edges_.size(),
                      from);

    // Initialize ghost vertices
    for (const auto &kv : ghost_pe) {
      cg.AddGhostVertex(kv.first, kv.second);
    }

    std::sort(edges_.begin(), edges_.end(), [&](const auto &left, const auto &right) {
        VertexID lhs_source = cg.GetLocalID(std::get<0>(left));
        VertexID lhs_target = cg.GetLocalID(std::get<1>(left));
        VertexID rhs_source = cg.GetLocalID(std::get<0>(right));
        VertexID rhs_target = cg.GetLocalID(std::get<1>(right));
        return (lhs_source < rhs_source
                  || (lhs_source == rhs_source && lhs_target < rhs_target));
    });

    for (auto &edge : edges_) {
      cg.AddEdge(cg.GetLocalID(std::get<0>(edge)), std::get<1>(edge), std::get<2>(edge));
    }

    cg.FinishConstruct();
    return cg;
  }

  PEID GetPEFromOffset(const VertexID v, 
                       std::vector<std::pair<VertexID, VertexID>> offset_array) const {
    for (PEID i = 0; i < offset_array.size(); ++i) {
      if (v >= offset_array[i].first && v < offset_array[i].second) {
        return i;
      }
    }
    return rank_;
  }
};

#endif
