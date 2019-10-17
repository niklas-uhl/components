/******************************************************************************
 * components.h *
 * Distributed computation of connected components
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

#ifndef _LOCAL_CONTRACTION_H_
#define _LOCAL_CONTRACTION_H_

#include <iostream>
#include <unordered_set>
#include <random>
#include <set>

#include "config.h"
#include "definitions.h"
#include "graph_io.h"
#include "dynamic_graph_access.h"
#include "static_graph_access.h"
#include "cag_builder.h"
#include "dynamic_contraction.h"
#include "utils.h"
#include "union_find.h"
#include "propagation.h"
#include "all_reduce.h"

class LocalContraction {
 public:
  LocalContraction(const Config &conf, const PEID rank, const PEID size)
      : rank_(rank),
        size_(size),
        config_(conf),
        iteration_(0) {}

  virtual ~LocalContraction() {
    delete local_contraction_;
    local_contraction_ = nullptr;
  };

  void FindComponents(DynamicGraphAccess &g, std::vector<VertexID> &g_labels) {
    rng_offset_ = size_ + config_.seed;
    if (config_.use_contraction) {
      FindLocalComponents(g, g_labels);

      CAGBuilder<DynamicGraphAccess> 
        first_contraction(g, g_labels, rank_, size_);
      DynamicGraphAccess cag = first_contraction.BuildDynamicComponentAdjacencyGraph();

      std::vector<VertexID> cag_labels(cag.GetNumberOfVertices(), 0);
      FindLocalComponents(cag, cag_labels);
      OutputStats<DynamicGraphAccess>(cag);

      CAGBuilder<DynamicGraphAccess>
        second_contraction(cag, cag_labels, rank_, size_);
      DynamicGraphAccess ccag = second_contraction.BuildDynamicComponentAdjacencyGraph();
      OutputStats<DynamicGraphAccess>(ccag);

      // TODO: Delete intermediate graph?
      // Keep contraction labeling for later
      local_contraction_ = new DynamicContraction(ccag, rank_, size_);

      PerformDecomposition(ccag);

      ApplyToLocalComponents(ccag, cag, cag_labels);
      ApplyToLocalComponents(cag, cag_labels, g, g_labels);
    } else {
      local_contraction_ = new DynamicContraction(g, rank_, size_);

      PerformDecomposition(g);

      g.ForallLocalVertices([&](const VertexID v) {
          g_labels[v] = g.GetVertexLabel(v);
      });
    }
  }

  void Output(DynamicGraphAccess &g) {
    g.OutputLabels();
  }

 private:
  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  // Algorithm state
  unsigned int iteration_;
  VertexID rng_offset_;

  // Statistics
  Timer iteration_timer_;

  // Contraction
  DynamicContraction *local_contraction_;

  void FindLocalComponents(DynamicGraphAccess &g, std::vector<VertexID> &label) {
    std::vector<bool> marked(g.GetNumberOfVertices(), false);
    std::vector<VertexID> parent(g.GetNumberOfVertices(), 0);

    g.ForallVertices([&](const VertexID v) {
      label[v] = g.GetGlobalID(v);
    });

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility<DynamicGraphAccess>::BFS(g, v, marked, parent);
    });

    // Set vertex label for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      label[v] = label[parent[v]];
    });
  }

  void FindLocalComponentsStatic(StaticGraphAccess &g, std::vector<VertexID> &label) {
    std::vector<bool> marked(g.GetNumberOfVertices(), false);
    std::vector<VertexID> parent(g.GetNumberOfVertices(), 0);

    g.ForallVertices([&](const VertexID v) {
      label[v] = g.GetGlobalID(v);
    });

    // Compute components
    g.ForallLocalVertices([&](const VertexID v) {
      if (!marked[v]) Utility<StaticGraphAccess>::BFS(g, v, marked, parent);
    });

    // Set vertex label for contraction
    g.ForallLocalVertices([&](const VertexID v) {
      label[v] = label[parent[v]];
    });
  }

  void PerformDecomposition(DynamicGraphAccess &g) {
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_timer_.Restart();
      iteration_++;
      if (global_vertices < config_.sequential_limit) 
        RunSequentialCC(g);
      else 
        RunContraction(g);
    }
    local_contraction_->UndoContraction();
  }

  void RunContraction(DynamicGraphAccess &g) {
    VertexID global_vertices = g.GatherNumberOfGlobalVertices();
    if (rank_ == ROOT) {
      if (iteration_ == 1)
        std::cout << "[STATUS] |-- Iteration " << iteration_ 
                  << " [TIME] " << "-" 
                  << " [ADD] " << global_vertices << std::endl;
      else
        std::cout << "[STATUS] |-- Iteration " << iteration_ 
                  << " [TIME] " << iteration_timer_.Elapsed() 
                  << " [ADD] " << global_vertices << std::endl;
    }
    iteration_timer_.Restart();

    // Draw uniform deviate per local vertex
    std::uniform_int_distribution<unsigned int> distribution(0, 99);
    std::mt19937
        generator(static_cast<unsigned int>(rank_ + config_.seed + iteration_ * rng_offset_));
    g.ForallLocalVertices([&](const VertexID v) {
      // Set preliminary deviate
      g.SetParent(v, g.GetGlobalID(v));
      g.SetVertexPayload(v, {static_cast<VertexID>(distribution(generator)),
                             g.GetVertexLabel(v), 
#ifdef TIEBREAK_DEGREE
                             g.GetVertexDegree(v),
#endif
                             g.GetVertexRoot(v)}, 
                         true);
#ifndef NDEBUG
      std::cout << "[R" << rank_ << ":" << iteration_ << "] update deviate "
                << g.GetGlobalID(v) << " -> " << g.GetVertexDeviate(v)
                << std::endl;
#endif
    });
    g.SendAndReceiveGhostVertices();

    // Perform update for local vertices
    // Find smallest label in N(v)
    std::vector<VertexPayload> n_smallest_neighbor(g.GetNumberOfLocalVertices());
    std::vector<VertexPayload> n_smallest_update(g.GetNumberOfLocalVertices());
    g.ForallLocalVertices([&](VertexID v) {
      n_smallest_neighbor[v] = g.GetVertexMessage(v);
      n_smallest_update[v] = g.GetVertexMessage(v);
      g.ForallNeighbors(v, [&](VertexID w) {
        // Store neighbor label
        if (g.GetVertexDeviate(w) < n_smallest_neighbor[v].deviate_ ||
            (g.GetVertexDeviate(w) == n_smallest_neighbor[v].deviate_ &&
                g.GetVertexLabel(w) < n_smallest_neighbor[v].label_)) {
          g.SetParent(v, g.GetGlobalID(w));
          n_smallest_neighbor[v] = {g.GetVertexDeviate(w), 
                                    g.GetVertexLabel(w),
#ifdef TIEBREAK_DEGREE
                                    g.GetVertexDegree(w),
#endif
                                    g.GetVertexRoot(w)};
          n_smallest_update[v] = {g.GetVertexDeviate(w), 
                                  g.GetVertexLabel(w),
#ifdef TIEBREAK_DEGREE
                                  g.GetVertexDegree(w),
#endif
                                  g.GetVertexRoot(w)};
        }
      });
    });

    g.ForallLocalVertices([&](VertexID v) {
      g.SetVertexPayload(v, std::move(n_smallest_update[v]));
    });

    // Receive variates
    g.SendAndReceiveGhostVertices();

    // Perform update for local vertices
    // Find smallest label in N(N(v))
    std::vector<VertexPayload> nn_smallest_neighbor(g.GetNumberOfLocalVertices());
    std::vector<VertexPayload> nn_smallest_update(g.GetNumberOfLocalVertices());
    g.ForallLocalVertices([&](VertexID v) {
      nn_smallest_neighbor[v] = g.GetVertexMessage(v);
      nn_smallest_update[v] = g.GetVertexMessage(v);
      g.ForallNeighbors(v, [&](VertexID w) {
        // Store neighbor label
        if (g.GetVertexDeviate(w) < nn_smallest_neighbor[v].deviate_ ||
            (g.GetVertexDeviate(w) == nn_smallest_neighbor[v].deviate_ &&
                g.GetVertexLabel(w) < nn_smallest_neighbor[v].label_)) {
          g.SetParent(v, g.GetGlobalID(w));
          nn_smallest_neighbor[v] = {g.GetVertexDeviate(w), 
                                     g.GetVertexLabel(w),
#ifdef TIEBREAK_DEGREE
                                     g.GetVertexDegree(w),
#endif
                                     g.GetVertexRoot(w)};
          nn_smallest_update[v] = {g.GetVertexDeviate(w), 
                                   g.GetVertexLabel(w),
#ifdef TIEBREAK_DEGREE
                                   g.GetVertexDegree(w),
#endif
                                   g.GetVertexRoot(w)};
        }
      });
    });

    g.ForallLocalVertices([&](VertexID v) {
      g.SetVertexPayload(v, std::move(nn_smallest_update[v]));
    });

    // Receive variates
    g.SendAndReceiveGhostVertices();

    // Determine remaining active vertices
    // TODO: Build shortcuts?
    local_contraction_->LocalContraction();

    OutputStats<DynamicGraphAccess>(g);

    // Count remaining number of vertices
    global_vertices = g.GatherNumberOfGlobalVertices();
    if (global_vertices > 0) {
      iteration_++;
      if (global_vertices < config_.sequential_limit) 
        RunSequentialCC(g);
      else 
        RunContraction(g);
    }
    local_contraction_->UndoContraction();
  }

  void ApplyToLocalComponents(DynamicGraphAccess &cag, 
                              DynamicGraphAccess &g, std::vector<VertexID> &g_label) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag.GetVertexLabel(cv);
    });
  }

  void ApplyToLocalComponents(DynamicGraphAccess &cag, 
                              std::vector<VertexID> &cag_label, 
                              DynamicGraphAccess &g, 
                              std::vector<VertexID> &g_label) {
    g.ForallLocalVertices([&](const VertexID v) {
      VertexID cv = cag.GetLocalID(g.GetContractionVertex(v));
      g_label[v] = cag_label[cv];
    });
  }

  void RunSequentialCC(DynamicGraphAccess &g) {
    // Build vertex mapping 
    google::dense_hash_map<VertexID, int> vertex_map; 
    vertex_map.set_empty_key(-1);
    std::vector<VertexID> reverse_vertex_map(g.GetNumberOfLocalVertices());
    int current_vertex = 0;
    g.ForallLocalVertices([&](const VertexID v) {
      vertex_map[v] = current_vertex;
      reverse_vertex_map[current_vertex++] = v;
    });

    // Init labels
    std::vector<VertexID> labels(g.GetNumberOfLocalVertices());
    for (VertexID i = 0; i < labels.size(); ++i) {
      labels[i] = g.GetVertexLabel(reverse_vertex_map[i]);
    }
    g.ForallLocalVertices([&](const VertexID v) {
      labels[v] = g.GetVertexLabel(v);
    });

    // Run all-reduce
    AllReduce<DynamicGraphAccess> ar(config_, rank_, size_);
    ar.FindComponents(g, labels);

    g.ForallLocalVertices([&](const VertexID v) {
      g.SetVertexLabel(v, labels[vertex_map[v]]);
    });
  }

  template <typename GraphType>
  void OutputStats(GraphType &g) {
    VertexID n = g.GatherNumberOfGlobalVertices();
    EdgeID m = g.GatherNumberOfGlobalEdges();

    // Determine min/maximum cut size
    EdgeID m_cut = g.GetNumberOfCutEdges();
    EdgeID min_cut, max_cut;
    MPI_Reduce(&m_cut, &min_cut, 1, MPI_VERTEX, MPI_MIN, ROOT,
               MPI_COMM_WORLD);
    MPI_Reduce(&m_cut, &max_cut, 1, MPI_VERTEX, MPI_MAX, ROOT,
               MPI_COMM_WORLD);

    if (rank_ == ROOT) {
      std::cout << "TEMP "
                << "s=" << config_.seed << ", "
                << "p=" << size_  << ", "
                << "n=" << n << ", "
                << "m=" << m << ", "
                << "c(min,max)=" << min_cut << "," << max_cut << std::endl;
    }
  }
};

#endif
