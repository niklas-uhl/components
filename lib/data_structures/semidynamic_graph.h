/******************************************************************************
 * graph_access.h
 *
 * Data structure for maintaining the (undirected) graph
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

#ifndef _SEMIDYNAMIC_GRAPH_H_
#define _SEMIDYNAMIC_GRAPH_H_

#include <mpi.h>

#include <algorithm>
#include <unordered_map>
#include <vector>
#include <memory>
#include <cmath>
#include <limits>
#include <stack>
#include <sstream>
#include <deque>
#include <tuple>
#include <unordered_set>
#include <boost/functional/hash.hpp>
#include <google/sparse_hash_set>
#include <google/dense_hash_set>
#include <google/sparse_hash_map>
#include <google/dense_hash_map>

#include "config.h"
#include "timer.h"

class SemidynamicGraph {
 public:
  SemidynamicGraph(const Config& conf, const PEID rank, const PEID size)
    : rank_(rank),
      size_(size),
      config_(conf),
      number_of_vertices_(0),
      number_of_local_vertices_(0),
      number_of_global_vertices_(0),
      number_of_edges_(0),
      number_of_cut_edges_(0),
      number_of_global_edges_(0),
      local_offset_(0),
      ghost_offset_(0),
      vertex_counter_(0),
      edge_counter_(0),
      local_duplicate_id_(0),
      global_duplicate_id_(0),
      ghost_counter_(0),
      comm_time_(0.0),
      send_volume_(0),
      recv_volume_(0) {
    label_shortcut_.set_empty_key(EmptyKey);
    label_shortcut_.set_deleted_key(DeleteKey);
    global_to_local_map_.set_empty_key(EmptyKey);
    global_to_local_map_.set_deleted_key(DeleteKey);
    // duplicates_.set_empty_key(EmptyKey);
    adjacent_pes_.set_empty_key(EmptyKey);
    adjacent_pes_.set_deleted_key(DeleteKey);
  }

  virtual ~SemidynamicGraph() {};

  //////////////////////////////////////////////
  // Graph construction
  //////////////////////////////////////////////
  void StartConstruct(VertexID local_n, VertexID ghost_n, VertexID local_offset) {
    number_of_local_vertices_ = local_n;
    number_of_vertices_ = local_n + ghost_n;

    // Overallocate
    if (config_.overallocate) {
      adjacent_edges_.reserve(1.2 * number_of_vertices_);
      local_vertices_data_.reserve(1.2 * local_n);
      ghost_vertices_data_.reserve(1.2 * ghost_n);
      parent_.reserve(1.2 * local_n);
      is_active_.reserve(1.2 * number_of_vertices_);
    }

    adjacent_edges_.resize(number_of_vertices_);
    local_vertices_data_.resize(local_n);
    ghost_vertices_data_.resize(ghost_n);

    local_offset_ = local_offset;
    ghost_offset_ = local_n;

    // Temp counter for properly counting new ghost vertices
    ghost_counter_ = local_n;

    parent_.resize(local_n);
    is_active_.resize(number_of_vertices_, true);
  }

  void FinishConstruct() { number_of_edges_ = edge_counter_; }

  //////////////////////////////////////////////
  // Graph iterators
  //////////////////////////////////////////////
  template<typename F>
  void ForallLocalVertices(F &&callback) {
    for (VertexID v = 0; v < GetLocalVertexVectorSize(); ++v) {
      if (IsActive(v)) callback(v);
    }
  }

  template<typename F>
  void ForallGhostVertices(F &&callback) {
    for (VertexID v = 0; v < GetGhostVertexVectorSize(); ++v) {
      if (IsActive(v + ghost_offset_)) callback(v + ghost_offset_);
    }
  }

  template<typename F>
  void ForallVertices(F &&callback) {
    ForallLocalVertices(callback);
    ForallGhostVertices(callback);
  }

  template<typename F>
  void ForallNeighbors(const VertexID v, F &&callback) {
    ForallAdjacentEdges(v, [&](EdgeID e) { callback(adjacent_edges_[v][e].target_); });
  }

  template<typename F>
  void ForallAdjacentEdges(const VertexID v, F &&callback) {
    for (EdgeID e = 0; e < GetVertexDegree(v); ++e) {
      callback(e);
    }
  }

  bool IsAdjacent(const VertexID source, const VertexID target) {
    bool adj = false;
    ForallNeighbors(source, [&](const VertexID w) {
      if (w == target) adj = true;
    });
    return adj;
  }

  inline bool IsActive(const VertexID v) const {
    return is_active_[v];
  }

  void SetActive(VertexID v, bool is_active) {
    if (is_active_[v] && !is_active) {
      number_of_local_vertices_ -= IsLocal(v);
      number_of_vertices_--;
    } else if (!is_active_[v] && is_active) {
      number_of_local_vertices_ += IsLocal(v);
      number_of_vertices_++;
    }
    is_active_[v] = is_active;
  }

  void SetAllVerticesActive(bool is_active) {
    for (VertexID v = 0; v < is_active_.size(); ++v) {
      SetActive(v, is_active);
    }
  }

  //////////////////////////////////////////////
  // Graph contraction
  //////////////////////////////////////////////
  inline void AllocateContractionVertices() {
    contraction_vertices_.resize(GetNumberOfVertices());
  }

  inline void SetContractionVertex(VertexID v, VertexID cv) {
    contraction_vertices_[v] = cv;
  }

  inline VertexID GetContractionVertex(VertexID v) const {
    return contraction_vertices_[v];
  }

  //////////////////////////////////////////////
  // Vertex mappings
  //////////////////////////////////////////////
  inline bool IsLocal(VertexID v) const {
    return v < GetLocalVertexVectorSize();
  }

  inline bool IsLocalFromGlobal(VertexID v) const {
    return v == global_duplicate_id_ 
            || (local_offset_ <= v && v < local_offset_ + GetLocalVertexVectorSize());
  }

  inline bool IsGhost(VertexID v) const {
    return global_to_local_map_.find(GetGlobalID(v))
        != global_to_local_map_.end();
  }


  inline bool IsGhostFromGlobal(VertexID v) const {
    return global_to_local_map_.find(v) != global_to_local_map_.end();
  }

  inline bool IsInterface(VertexID v) const {
    return local_vertices_data_[v].is_interface_vertex_;
  }

  inline bool IsInterfaceFromGlobal(VertexID v) const {
    return local_vertices_data_[GetLocalID(v)].is_interface_vertex_;
  }

  inline VertexID GetLocalID(VertexID v) const {
    if (IsLocalFromGlobal(v)) {
      if (global_duplicate_id_ == v) {
        return local_duplicate_id_;
      } else {
        return v - local_offset_;
      }
    } else {
      return global_to_local_map_.find(v)->second;
    }
  }

  inline VertexID GetGlobalID(VertexID v) const {
    if (IsLocal(v)) {
      if (local_duplicate_id_ == v) {
        return global_duplicate_id_;
      } else {
        return v + local_offset_;
      }
    } else {
      return ghost_vertices_data_[v - ghost_offset_].global_id_;
    }
  }

  inline PEID GetPE(VertexID v) const {
    return IsLocal(v) ? rank_
                      : ghost_vertices_data_[v - ghost_offset_].rank_;

  }

  //////////////////////////////////////////////
  // Manage local vertices/edges
  //////////////////////////////////////////////
  inline VertexID GetNumberOfVertices() const { return number_of_vertices_; }
  inline VertexID GetVertexVectorSize() const { return GetLocalVertexVectorSize() + GetGhostVertexVectorSize(); }

  inline VertexID GetNumberOfGlobalVertices() const { return number_of_global_vertices_; }

  inline VertexID GetNumberOfGlobalEdges() const { return number_of_global_edges_; }

  inline VertexID GetLocalOffset() const {
    return local_offset_;
  }

  inline VertexID GetNumberOfLocalVertices() const { return number_of_local_vertices_; }
  inline VertexID GetLocalVertexVectorSize() const { return local_vertices_data_.size(); }

  inline VertexID GetNumberOfGhostVertices() const { return number_of_vertices_ - number_of_local_vertices_; }
  inline VertexID GetGhostVertexVectorSize() const { return ghost_vertices_data_.size(); }

  inline EdgeID GetNumberOfEdges() const { return number_of_edges_; }

  inline EdgeID GetNumberOfCutEdges() const { return number_of_cut_edges_; }

  inline void ResetNumberOfCutEdges() { number_of_cut_edges_ = 0; }

  VertexID GatherNumberOfGlobalVertices(bool force=true) {
    if (number_of_global_vertices_ == 0 || force) {
      number_of_global_vertices_ = 0;
#ifndef NDEBUG
      VertexID local_vertices = 0;
      ForallLocalVertices([&](const VertexID v) { local_vertices++; });
      if (local_vertices != number_of_local_vertices_) {
        std::cout << "This shouldn't happen (different number of vertices local=" << local_vertices << ", counter=" << number_of_local_vertices_ << ", datasize=" << GetLocalVertexVectorSize() << ")" << std::endl;
        exit(1);
      }
#endif
      // Check if all PEs are done
      comm_timer_.Restart();
      MPI_Allreduce(&number_of_local_vertices_,
                    &number_of_global_vertices_,
                    1,
                    MPI_VERTEX,
                    MPI_SUM,
                    MPI_COMM_WORLD);
      comm_time_ += comm_timer_.Elapsed();
    }
    return number_of_global_vertices_;
  }

  VertexID GatherNumberOfGlobalEdges(bool force=true) {
    if (number_of_global_edges_ == 0 || force) {
      number_of_global_edges_ = 0;
// #ifndef NDEBUG
//       VertexID local_edges = 0;
//       ForallVertices([&](const VertexID v) { 
//           ForallNeighbors(v, [&](const VertexID w) { local_edges++; });
//       });
//       if (local_edges != number_of_edges_) {
//         std::cout << "This shouldn't happen (different number of edges local=" << local_edges << ", counter=" << number_of_edges_ << ")" << std::endl;
//         exit(1);
//       }
// #endif
      // Check if all PEs are done
      comm_timer_.Restart();
      MPI_Allreduce(&number_of_edges_,
                    &number_of_global_edges_,
                    1,
                    MPI_VERTEX,
                    MPI_SUM,
                    MPI_COMM_WORLD);
      comm_time_ += comm_timer_.Elapsed();
      number_of_global_edges_ /= 2;
    }
    return number_of_global_edges_;
  }

  void SetParent(const VertexID v, const VertexID parent_v) {
    parent_[v] = parent_v;
  }

  inline VertexID GetParent(const VertexID v) {
    return parent_[v];
  }

  inline VertexID AddVertex() { return vertex_counter_++; }

  VertexID AddGhostVertex(VertexID v, PEID pe) {
    global_to_local_map_[v] = ghost_counter_;

    // Fix overflows
    if (vertex_counter_ >= is_active_.size()) {
      is_active_.resize(vertex_counter_ + 1);
    }
    if (ghost_counter_ - ghost_offset_ >= ghost_vertices_data_.size()) {
      ghost_vertices_data_.resize(ghost_counter_ - ghost_offset_ + 1);
    }

    // Update data
    ghost_vertices_data_[ghost_counter_ - ghost_offset_].rank_ = pe;
    ghost_vertices_data_[ghost_counter_ - ghost_offset_].global_id_ = v;
    is_active_[vertex_counter_] = true;

    vertex_counter_++;
    return ghost_counter_++;
  }

  // TODO: Not sure if this works
  inline void AddDuplicateVertex(VertexID global_id) {
    local_duplicate_id_ = 0;
    global_duplicate_id_ = global_id;
  }

  EdgeID AddEdge(VertexID from, VertexID to, PEID rank) {
    if (IsLocalFromGlobal(to)) {
      AddLocalEdge(from, to);
    } else {
#ifndef NDEBUG
      if (rank == size_) {
        throw "This shouldn't happen";
      }
      if (IsGhostFromGlobal(to)) { // true if ghost already in map, otherwise false
        local_vertices_data_[from].is_interface_vertex_ = true;
        number_of_cut_edges_++;
        AddGhostEdge(from, to);
        SetAdjacentPE(rank, true);
      } else {
        //std::cout << "This shouldn't happen" << std::endl;
        throw "This shouldn't happen";
      }
#else 
      local_vertices_data_[from].is_interface_vertex_ = true;
      number_of_cut_edges_++;
      AddGhostEdge(from, to);
      SetAdjacentPE(rank, true);
#endif
    }
    edge_counter_++;
    return edge_counter_;
  }

  void AddLocalEdge(VertexID from, VertexID to) {
    if (from >= adjacent_edges_.size()) {
      adjacent_edges_.resize(from + 1);
    }
    adjacent_edges_[from].emplace_back(to - local_offset_);
  }

  void AddGhostEdge(VertexID from, VertexID to) {
    if (from >= adjacent_edges_.size()) {
      adjacent_edges_.resize(from + 1);
    }
    adjacent_edges_[from].emplace_back(global_to_local_map_[to]);
  }


  void ReserveEdgesForVertex(VertexID v, VertexID num_edges) {
    adjacent_edges_[v].reserve(num_edges);
  }

  void RemoveAllEdges(VertexID from) {
    ForallNeighbors(from, [&](const VertexID w) {
      if (IsGhost(w)) number_of_cut_edges_--;
      number_of_edges_--;
    });
    adjacent_edges_[from].clear();
  }

  // Local IDs
  bool IsConnected(VertexID from, VertexID to) {
    ForallNeighbors(from, [&](VertexID v) {
        if (v == to) return true; 
    });
    return false;
  }

  inline VertexID GetVertexDegree(const VertexID v) const {
    return adjacent_edges_[v].size();
  }

  //////////////////////////////////////////////
  // Manage adjacent PEs
  //////////////////////////////////////////////
  inline PEID GetNumberOfAdjacentPEs() const {
    return adjacent_pes_.size();
  }

  template<typename F>
  void ForallAdjacentPEs(F &&callback) {
    for (const PEID &pe : adjacent_pes_) {
      callback(pe);
    }
  }

  inline bool IsAdjacentPE(const PEID pe) const {
    return adjacent_pes_.find(pe) != adjacent_pes_.end();
  }

  void SetAdjacentPE(const PEID pe, const bool is_adj) {
    if (pe == rank_) return;
    if (is_adj) {
      if (IsAdjacentPE(pe)) return;
      else adjacent_pes_.insert(pe);
    } else {
      if (!IsAdjacentPE(pe)) return;
      else adjacent_pes_.erase(pe);
    }
  }

  void ResetAdjacentPEs() {
    adjacent_pes_.clear();
  }

  //////////////////////////////////////////////
  // I/O
  //////////////////////////////////////////////
  bool CheckDuplicates() {
    // google::dense_hash_set<VertexID> neighbors;
    ForallLocalVertices([&](const VertexID v) {
      std::unordered_set<VertexID> neighbors;
      ForallNeighbors(v, [&](const VertexID w) {
        if (neighbors.find(w) != end(neighbors)) {
          std::cout << "[R" << rank_ << ":0] DUPL (" << GetGlobalID(v) << "," << GetGlobalID(w) << "[" << GetPE(w) << "])" << std::endl;
          return true;
        }
        neighbors.insert(w);
      });
    });
    return false;
  }

  void OutputLocal() {
    ForallLocalVertices([&](const VertexID v) {
      std::stringstream out;
      out << "[R" << rank_ << "] [V] "
          << GetGlobalID(v) << " (local_id=" << v
           << ", pe=" << rank_ << ")";
      std::cout << out.str() << std::endl;
    });

    ForallLocalVertices([&](const VertexID v) {
      std::stringstream out;
      out << "[R" << rank_ << "] [N] "
          << GetGlobalID(v) << " -> ";
      ForallNeighbors(v, [&](VertexID u) {
        out << GetGlobalID(u) << " (local_id=" << u
            << ", pe=" << GetPE(u) << ") ";
      });
      std::cout << out.str() << std::endl;
    });
  }

  void OutputGhosts() {
    PEID rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::cout << "[R" << rank << "] [G] [ ";
    for (auto &e : global_to_local_map_) {
      std::cout << e.first << " ";
    }
    std::cout << "]" << std::endl;
  }

  void OutputComponents(std::vector<VertexID> &labels) {
    VertexID global_num_vertices = GatherNumberOfGlobalVertices();
    // Gather component sizes
    google::dense_hash_map<VertexID, VertexID> local_component_sizes; 
    local_component_sizes.set_empty_key(EmptyKey);
    local_component_sizes.set_deleted_key(DeleteKey);
    ForallLocalVertices([&](const VertexID v) {
      if (GetGlobalID(v) <= global_num_vertices) {
        VertexID c = labels[v];
        if (local_component_sizes.find(c) == end(local_component_sizes))
          local_component_sizes[c] = 0;
        local_component_sizes[c]++;
      }
    });

    // Gather component message
    std::vector<std::pair<VertexID, VertexID>> local_components;
    // local_components.reserve(local_component_sizes.size());
    for(auto &kv : local_component_sizes)
      local_components.emplace_back(kv.first, kv.second);
    // [MEMORY]: Might be too small
    int num_local_components = local_components.size();

    // Exchange number of local components
    std::vector<int> num_components(size_);
    MPI_Gather(&num_local_components, 1, MPI_INT, &num_components[0], 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    // Compute diplacements
    std::vector<int> displ_components(size_, 0);
    // [MEMORY]: Might be too small
    int num_global_components = 0;
    for (PEID i = 0; i < size_; ++i) {
      displ_components[i] = num_global_components;
      num_global_components += num_components[i];
    }

    // Add datatype
    MPI_Datatype MPI_COMP;
    MPI_Type_vector(1, 2, 0, MPI_VERTEX, &MPI_COMP);
    MPI_Type_commit(&MPI_COMP);

    // Exchange components
    std::vector<std::pair<VertexID, VertexID>> global_components(num_global_components);
    MPI_Gatherv(&local_components[0], num_local_components, MPI_COMP,
                &global_components[0], &num_components[0], &displ_components[0], MPI_COMP,
                ROOT, MPI_COMM_WORLD);

    if (rank_ == ROOT) {
      google::dense_hash_map<VertexID, VertexID> global_component_sizes; 
      global_component_sizes.set_empty_key(EmptyKey);
      global_component_sizes.set_deleted_key(DeleteKey);
      for (auto &comp : global_components) {
        VertexID c = comp.first;
        VertexID size = comp.second;
        if (global_component_sizes.find(c) == end(global_component_sizes))
          global_component_sizes[c] = 0;
        global_component_sizes[c] += size;
      }

      google::dense_hash_map<VertexID, VertexID> condensed_component_sizes; 
      condensed_component_sizes.set_empty_key(EmptyKey);
      condensed_component_sizes.set_deleted_key(DeleteKey);
      for (auto &cs : global_component_sizes) {
        VertexID size = cs.second;
        if (condensed_component_sizes.find(size) == end(condensed_component_sizes)) {
          condensed_component_sizes[size] = 0;
        }
        condensed_component_sizes[size]++;
      }

      // Build final vector
      std::vector<std::pair<VertexID, VertexID>> components;
      components.reserve(condensed_component_sizes.size());
      for(auto &kv : condensed_component_sizes)
        components.emplace_back(kv.first, kv.second);
      std::sort(begin(components), end(components));

      // std::cout << "COMPONENTS [ ";
      // for (auto &comp : components)
      //   std::cout << "size=" << comp.first << " (num=" << comp.second << ") ";
      // std::cout << "]" << std::endl;

      VertexID total_num_no_isolated = 0;
      for (auto &comp : components)
        if (comp.first != 1) total_num_no_isolated += comp.second;
      std::cout << "NUM COMPONENTS " << total_num_no_isolated << std::endl;
    }
  }

  void Logging(bool active);

  inline float GetCommTime() {
    return comm_time_;
  }

  inline VertexID GetSendVolume() {
    return send_volume_;
  }

  inline VertexID GetReceiveVolume() {
    return recv_volume_;
  }

  inline void ResetCommTime() {
    comm_time_ = 0.0;
  }

  inline void ResetSendVolume() {
    send_volume_ = 0;
  }

  inline void ResetReceiveVolume() {
    recv_volume_ = 0;
  }

 protected:
  // Structs
  struct Vertex {
    EdgeID first_edge_;

    Vertex() : first_edge_(std::numeric_limits<EdgeID>::max()) {}
    explicit Vertex(EdgeID e) : first_edge_(e) {}
  };

  struct LocalVertexData {
    bool is_interface_vertex_;

    LocalVertexData()
        : is_interface_vertex_(false) {}
    LocalVertexData(const VertexID id, bool interface)
        : is_interface_vertex_(interface) {}
  };

  struct GhostVertexData {
    PEID rank_;
    VertexID global_id_;

    GhostVertexData()
        : rank_(0), global_id_(0) {}
    GhostVertexData(PEID rank, VertexID global_id)
        : rank_(rank), global_id_(global_id) {}
  };

  struct Edge {
    VertexID target_;

    Edge() : target_(0) {}
    explicit Edge(VertexID target) : target_(target) {}
  };


  // Network information
  PEID rank_, size_;

  // Configuration
  Config config_;

  // Vertices and edges
  std::vector<std::vector<Edge>> adjacent_edges_;

  std::vector<LocalVertexData> local_vertices_data_;
  std::vector<GhostVertexData> ghost_vertices_data_;

  // Shortcutting
  std::vector<VertexID> parent_;
  google::dense_hash_map<VertexID, VertexID> label_shortcut_;

  VertexID number_of_vertices_;
  VertexID number_of_local_vertices_;
  VertexID number_of_global_vertices_;

  EdgeID number_of_edges_;
  EdgeID number_of_cut_edges_;
  EdgeID number_of_global_edges_;

  // Vertex mapping
  VertexID local_offset_;
  VertexID ghost_offset_;
  google::dense_hash_map<VertexID, VertexID> global_to_local_map_;

  // Contraction
  std::vector<VertexID> contraction_vertices_;
  std::vector<bool> is_active_;

  // Duplicates
  // google::dense_hash_set<VertexID> duplicates_;
  VertexID local_duplicate_id_;
  VertexID global_duplicate_id_;

  // Adjacent PEs
  google::dense_hash_set<PEID> adjacent_pes_;

  // Temporary counters
  VertexID vertex_counter_;
  VertexID ghost_counter_;
  EdgeID edge_counter_;

  // Statistics
  float comm_time_;
  Timer comm_timer_;
  VertexID send_volume_;
  VertexID recv_volume_;
};

#endif
