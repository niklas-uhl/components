/******************************************************************************
 * utils.h
 *
 * Utility algorithms needed for connected components
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

#ifndef _UTILITY_H_
#define _UTILITY_H_

#include <queue>

#include "graph_access.h"

class Utility {
 public:
  static void BFS(GraphAccess &g,
                  const VertexID &start,
                  std::vector<bool> &marked,
                  std::vector<VertexID> &parent) {
    // Standard BFS
    std::queue<VertexID> q;
    q.push(start);
    while (!q.empty()) {
      VertexID v = q.front();
      q.pop();
      parent[v] = start;
      marked[v] = true;
      g.ForallNeighbors(v, [&](VertexID w) {
        if (g.IsLocal(w) && !marked[w]) {
          q.push(w);
        }
      });
    }
  }
};

#endif