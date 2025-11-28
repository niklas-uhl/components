/******************************************************************************
 * ghost_communicator.h
 *
 * Communication patterns for ghost vertices in distributed graph.
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

#ifndef _VERTEX_COMMUNICATOR_H_
#define _VERTEX_COMMUNICATOR_H_

#include <memory>
#include <iostream>
#include <random>

#include "kacomp/config.h"
#include "kacomp/definitions.h"
#include "kacomp/communication/payload.h"
// #include "dynamic_graph_comm.h"
// #include "semidynamic_graph_comm.h"
// #include "static_graph_comm.h"

#include "kacomp/communication/comm_utils.h"
#include <google/dense_hash_map>
#include <google/dense_hash_set>
#include <google/sparse_hash_set>

namespace kacomp {

template<typename GraphType>
class VertexCommunicator {
 public:
  VertexCommunicator(const Config &conf, const PEID rank, const PEID size)
      : g_(nullptr),
        use_sampling_(false),
        rank_(rank),
        size_(size),
        config_(conf),
        comm_time_(0.0),
        send_volume_(0),
        recv_volume_(0) {
    packed_pes_.set_empty_key(EmptyKey);
    packed_pes_.set_deleted_key(DeleteKey);
    send_buffers_.set_empty_key(EmptyKey);
    send_buffers_.set_deleted_key(DeleteKey);
    receive_buffers_.set_empty_key(EmptyKey);
    receive_buffers_.set_deleted_key(DeleteKey);
    neighborhood_sample_.set_empty_key(EmptyKey);
    neighborhood_sample_.set_deleted_key(DeleteKey);
    message_tag_ = static_cast<unsigned int>(CommTag);
  }
  virtual ~VertexCommunicator() {};

  VertexCommunicator(const VertexCommunicator &rhs) = default;
  VertexCommunicator(VertexCommunicator &&rhs) = default;
 
  inline void SetGraph(GraphType *g) {
    g_ = g;
  }

  inline bool IsPackedPE(const PEID pe) const {
    return packed_pes_.find(pe) != packed_pes_.end();
  }

  void SetPackedPE(const PEID pe, const bool is_packed) {
    if (pe == rank_) return;
    if (is_packed) {
      if (IsPackedPE(pe)) return;
      else packed_pes_.insert(pe);
    } else {
      if (!IsPackedPE(pe)) return;
      else packed_pes_.erase(pe);
    }
  }

  void AddMessage(VertexID v, const VertexPayload &msg);

  void SampleVertexNeighborhood(const VertexID &v, const float sampling_factor);

  void UpdateGhostVertices();

  void SendAndReceiveGhostVertices() {
    comm_time_ += CommunicationUtility::AllToAll(send_buffers_, receive_buffers_, 
                                                 rank_, size_, message_tag_, config_.use_regular);
    message_tag_++;
    send_volume_ += CommunicationUtility::ClearBuffers(send_buffers_);
    UpdateGhostVertices();
    recv_volume_ += CommunicationUtility::ClearBuffers(receive_buffers_);
  }

  inline float GetCommTime() {
    return comm_time_;
  }

  inline VertexID GetSendVolume() {
    return send_volume_;
  }

  inline VertexID GetReceiveVolume() {
    return recv_volume_;
  }

 private:
  GraphType *g_;

  PEID rank_, size_;

  google::dense_hash_set<PEID> packed_pes_;
  google::dense_hash_map<PEID, VertexBuffer> send_buffers_;
  google::dense_hash_map<PEID, VertexBuffer> receive_buffers_;

  // Configuration
  Config config_;

  // Neighborhood sampling
  bool use_sampling_;
  google::dense_hash_map<VertexID, google::sparse_hash_set<VertexID>> neighborhood_sample_;

  VertexID message_tag_;

  float comm_time_;
  VertexID send_volume_;
  VertexID recv_volume_;

  void PlaceInBuffer(const PEID &pe,
                     const VertexID &v,
                     const VertexPayload &msg);
};

template<typename GraphType>
void VertexCommunicator<GraphType>::AddMessage(const VertexID v,
                                               const VertexPayload &msg) {
  if (use_sampling_) {
    if (neighborhood_sample_.find(v) != neighborhood_sample_.end()) {
      for (const VertexID &u : neighborhood_sample_[v]) {
        if (!g_->IsLocal(u)) {
          PEID neighbor = g_->GetPE(u);
          if (!IsPackedPE(neighbor)) {
            PlaceInBuffer(neighbor, v, msg);
          }
        }
      }
    }
  } else {
    g_->ForallNeighbors(v, [&](const VertexID u) {
      if (!g_->IsLocal(u)) {
        PEID neighbor = g_->GetPE(u);
        if (!IsPackedPE(neighbor)) {
          PlaceInBuffer(neighbor, v, msg);
        }
      }
    });
  }

  g_->ForallNeighbors(v, [&](const VertexID u) {
    if (!g_->IsLocal(u)) SetPackedPE(g_->GetPE(u), false);
  });
}

template<typename GraphType>
void VertexCommunicator<GraphType>::SampleVertexNeighborhood(const VertexID &v,
                                                             const float sampling_factor) {
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<> dist(0, 1);
  g_->ForallNeighbors(v, [&](const VertexID u) {
    if (dist(mt) < sampling_factor) {
      neighborhood_sample_[v].insert(u);
    }
  });
  use_sampling_ = true;
}

template<typename GraphType>
void VertexCommunicator<GraphType>::UpdateGhostVertices() {
  for (const auto &kv : receive_buffers_) {
    const auto &buffer = kv.second;
#ifdef TIEBREAK_DEGREE
    for (VertexID i = 0; i < buffer.size(); i += 5) {
#else 
    for (VertexID i = 0; i < buffer.size(); i += 4) {
#endif
      VertexID global_id = buffer[i];
      VertexID deviate = buffer[i + 1];
      VertexID label = buffer[i + 2];
      PEID root = static_cast<PEID>(buffer[i + 3]);
#ifdef TIEBREAK_DEGREE
      VertexID degree = buffer[i + 4];
#endif
      if (!g_->IsGhostFromGlobal(global_id) || !g_->IsLocalFromGlobal(global_id)) {
	continue;
      }
      g_->HandleGhostUpdate(g_->GetLocalID(global_id), 
                            label, 
                            deviate, 
#ifdef TIEBREAK_DEGREE
                            degree,
#endif
                            root);
    }
  }
}

template<typename GraphType>
void VertexCommunicator<GraphType>::PlaceInBuffer(const PEID &pe, 
                                                  const VertexID &v,
                                                  const VertexPayload &msg) {
    // Unpack msg and add content (Sender, Deviate, Component, PE (of component))
    send_buffers_[pe].emplace_back(g_->GetGlobalID(v));
    send_buffers_[pe].emplace_back(msg.deviate_);
    send_buffers_[pe].emplace_back(msg.label_);
    send_buffers_[pe].emplace_back(msg.root_);
#ifdef TIEBREAK_DEGREE
    send_buffers_[pe].emplace_back(msg.degree_);
#endif
    SetPackedPE(pe, true);
}


}


// template class VertexCommunicator<DynamicGraphCommunicator>;
// template class VertexCommunicator<SemidynamicGraphCommunicator>;
// template class VertexCommunicator<StaticGraphCommunicator>;

#endif
