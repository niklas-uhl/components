/******************************************************************************
 * io_utils.h
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

#ifndef _IO_UTILITY_H_
#define _IO_UTILITY_H_

#include "config.h"
#include "definitions.h"
#include "io/graph_io.h"
#include <kagen.h>

class IOUtility {
 public:
  template<typename GraphType>
  static void LoadGraph(GraphType &g, 
                        Config &config,
                        MPI_Comm comm) {
    // File I/O
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (config.input_type == "file") {
      GraphIO::ReadMETISFile<GraphType>(g, config, rank, size, MPI_COMM_WORLD);
    } else if (config.input_type == "edgefile") {
      GraphIO::ReadSortedEdgeFile<GraphType>(g, config, rank, size, MPI_COMM_WORLD);
    } else if (config.input_type == "partition") {
      GraphIO::ReadPartitionedSortedEdgeFile<GraphType>(g, config, rank, size, MPI_COMM_WORLD);
    } else if (config.input_type == "binary") {
      GraphIO::ReadSortedBinaryFile<GraphType>(g, config, rank, size, MPI_COMM_WORLD);
    } else if (config.gen != "null") {
      // Generator I/O
      kagen::KaGen gen(comm);
      gen.EnableBasicStatistics();
      kagen::EdgeList edge_list;
      auto result = GenerateSyntheticGraph(gen, config);
      GraphIO::ReadMETISGenerator<GraphType>(g, config, rank, size, MPI_COMM_WORLD, result);
    } else {
      std::cout << "I/O type not supported" << std::endl;
      MPI_Finalize();
      exit(1);
    }
  }

  static kagen::KaGenResult GenerateSyntheticGraph(kagen::KaGen &gen,
                                     Config &config) {
      gen.SetSeed(config.seed);
      if (config.gen == "gnm_undirected")
          return gen.GenerateUndirectedGNM(config.gen_n, config.gen_m);
      else if (config.gen == "rdg_2d")
          return gen.GenerateRDG2D(config.gen_n, config.gen_periodic);
      else if (config.gen == "rdg_3d")
          return gen.GenerateRDG3D(config.gen_n);
      else if (config.gen == "rgg_2d")
          return gen.GenerateRGG2D(config.gen_n, config.gen_r);
      else if (config.gen == "rgg_3d")
          return gen.GenerateRGG3D(config.gen_n, config.gen_r);
      else if (config.gen == "rhg")
          return gen.GenerateRHG(config.gen_gamma, config.gen_n, config.gen_d);
      else if (config.gen == "ba")
          return gen.GenerateBA(config.gen_n, config.gen_d);
      else if (config.gen == "grid_2d")
          return gen.GenerateGrid2D(config.gen_n, config.gen_m, config.gen_p, config.gen_periodic);
      else {
        throw "Generator not supported";
      }
  }

  template<typename GraphType>
  static void PrintGraphParams(GraphType &g,
                               Config &config,
                               PEID rank, PEID size) {
    VertexID n_local = g.GetNumberOfLocalVertices();
    EdgeID m_local = g.GetNumberOfEdges()/2;
    VertexID n_global = g.GatherNumberOfGlobalVertices();
    EdgeID m_global = g.GatherNumberOfGlobalEdges();

    VertexID highest_degree = 0;
    g.ForallLocalVertices([&](const VertexID v) {
      if (g.GetVertexDegree(v) > highest_degree) {
        highest_degree = g.GetVertexDegree(v);
      }
    });

    // Determine min/maximum cut size
    EdgeID cut_local = g.GetNumberOfCutEdges();
    EdgeID min_cut, max_cut;
    MPI_Reduce(&cut_local, &min_cut, 1, MPI_VERTEX, MPI_MIN, ROOT,
               MPI_COMM_WORLD);
    MPI_Reduce(&cut_local, &max_cut, 1, MPI_VERTEX, MPI_MAX, ROOT,
               MPI_COMM_WORLD);

    if (config.print_verbose) {
      std::cout << "LOCAL INPUT" 
                << " rank=" << rank
                << " n=" << n_local 
                << " m=" << m_local 
                << " c=" << cut_local 
                << " max_d=" << highest_degree << std::endl;
    }
    if (rank == ROOT) {
      std::cout << "GLOBAL INPUT"
                << " s=" << config.seed
                << " p=" << size
                << " n=" << n_global
                << " m=" << m_global
                << " c(min,max)=" << min_cut << "," << max_cut << std::endl;
    }
  }
};

#endif
