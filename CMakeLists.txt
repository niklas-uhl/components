cmake_minimum_required(VERSION 3.28)
project(components LANGUAGES CXX)

find_package(MPI REQUIRED)

#set(KACOMP_DEFINITIONS "")
option(KACOMP_TIEBREAK_DEGREE "Tiebreak based on degree instead of ID." OFF)
option(KACOMP_MEMINFO "Show memory consumed." OFF)
option(KACOMP_USE_CONNECT_IO "Use connect IO for reading files" OFF)

if(KACOMP_TIEBREAK_DEGREE)
    list(APPEND KACOMP_DEFINITIONS "-DTIEBREAK_DEGREE")
endif()
if(KACOMP_MEMINFO)
    list(APPEND KACOMP_DEFINITIONS "-DMEMINFO")
endif()
if(KACOMP_USE_CONNECT_IO)
    list(APPEND KACOMP_DEFINITIONS "-DUSE_CONNECT_IO")
endif()
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${KACOMP_DEFINITIONS}")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS} -std=c++14 -fno-omit-frame-pointer -march=native -DNDEBUG -DNSTATUS -DNWARMUP -Wall -fPIC -O3 -g")
list(APPEND KACOMP_DEFINITIONS "-DNSTATUS" "-DNWARMUP")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS} -fsanitize=address -std=c++14 -fno-omit-frame-pointer -march=native -fno-stack-limit -Wall -fPIC -Og -g")
#set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")

include(FetchContent)


# this needs to be set as a cache variable, since IPS4o uses CMake < 3.13 and therefore does not respect
# CMake policy CMP0077 https://cmake.org/cmake/help/latest/policy/CMP0077.html
set(IPS4O_DISABLE_PARALLEL ON CACHE BOOL "Disable shared memory parallelism of IPS4o")
FetchContent_Declare(
  ips4o
  GIT_REPOSITORY https://github.com/ips4o/ips4o
  GIT_TAG 08a5b92
  SYSTEM
  EXCLUDE_FROM_ALL
)
set(KAGEN_BUILD_APPS OFF)
set(KAGEN_BUILD_EXAMPLES OFF)
set(KAGEN_USE_CGAL OFF)
set(KAGEN_USE_SPARSEHASH OFF)
set(KAGEN_USE_XXHASH ON)
FetchContent_Declare(
  kagen
  GIT_REPOSITORY https://github.com/KarlsruheGraphGeneration/KaGen
  GIT_TAG main #v1.2.1
  SYSTEM
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(ips4o kagen)
list(APPEND CMAKE_MODULE_PATH ${kagen_SOURCE_DIR}/cmake/modules)
find_package(Sparsehash REQUIRED)

# add_subdirectory(extern/KaHIP)


#include_directories("${PROJECT_SOURCE_DIR}/extern/mxx/include")
#include_directories("${PROJECT_SOURCE_DIR}/extern")

#include_directories("${PROJECT_SOURCE_DIR}/extern/bliss/src/")
#include_directories("${PROJECT_SOURCE_DIR}/extern/bliss/ext/")

#include our build directory for bliss config files
#include_directories("${CMAKE_BINARY_DIR}")


set(KACOMP_INCLUDE_DIRS
        "${CMAKE_CURRENT_LIST_DIR}/app/"
        "${CMAKE_CURRENT_LIST_DIR}/lib/"
        "${CMAKE_CURRENT_LIST_DIR}/lib/communication/"
        "${CMAKE_CURRENT_LIST_DIR}/lib/components/"
        "${CMAKE_CURRENT_LIST_DIR}/lib/contraction/"
        "${CMAKE_CURRENT_LIST_DIR}/lib/data_structures/"
        "${CMAKE_CURRENT_LIST_DIR}/lib/io/"
        "${CMAKE_CURRENT_LIST_DIR}/lib/tools/"
          # extern/kagen/extlib/sampling/sampling
          # extern/kagen/include
          # extern/kagen/include/generators
          # extern/kagen/include/generators/barabassi
          # extern/kagen/include/generators/geometric
          # extern/kagen/include/generators/geometric/delaunay
          # extern/kagen/include/generators/geometric/rgg
          # extern/kagen/include/generators/gnm
          # extern/kagen/include/generators/gnp
          # extern/kagen/include/generators/hyperbolic
          # extern/kagen/include/io
          # extern/kagen/include/tools
          # extern/kagen/interface/
)


set(KACOMP_SOURCES
          # app
          # lib
          # lib/communication
          lib/communication/vertex_comm.cpp
          # lib/components
          # lib/contraction
          # lib/data_structures
          lib/data_structures/dynamic_graph_comm.cpp
          lib/data_structures/semidynamic_graph_comm.cpp
          lib/data_structures/static_graph_comm.cpp
          # lib/io
          # lib/tools
)


add_library(KaCC ${KACOMP_SOURCES})
target_include_directories(KaCC PUBLIC "${KACOMP_INCLUDE_DIRS}")
target_compile_definitions(KaCC PUBLIC ${KACOMP_DEFINITIONS})
target_compile_features(KaCC PUBLIC cxx_std_17)

# add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/extern/ips4o" EXCLUDE_FROM_ALL)
# add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/extern/kagen" EXCLUDE_FROM_ALL)

# set(BUILD_SHARED_LIBS OFF CACHE BOOL "Foo")
# add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/extern/bliss")
# add_library(bliss INTERFACE)
# target_include_directories(bliss INTERFACE "${CMAKE_CURRENT_LIST_DIR}/extern/bliss/retired/src/")   #BLISS cmake needs to be executed
# target_include_directories(bliss INTERFACE "${CMAKE_CURRENT_LIST_DIR}/extern/bliss/src")
# target_include_directories(bliss INTERFACE "${CMAKE_CURRENT_LIST_DIR}/extern/bliss/ext/mxx/include/")   #BLISS cmake needs to be executed
# target_include_directories(bliss INTERFACE "${CMAKE_CURRENT_BINARY_DIR}")
set(LIBS
        # sampling
        MPI::MPI_CXX
        KaGen::KaGen
        ips4o
	Sparsehash::Sparsehash
        # bliss
        #kahip
        # asan
        )

target_link_libraries(KaCC PUBLIC ${LIBS})
# target_link_libraries(KaCC PUBLIC backward_object backward)

add_executable(exponential "app/exp_contraction.cpp")
target_link_libraries(exponential PUBLIC KaCC)
add_executable(allreduce "app/all_reduce.cpp")
target_link_libraries(allreduce PUBLIC KaCC)
# add_executable(local ${SOURCES} "app/local_contraction.cpp")
# add_executable(shortcuts ${SOURCES} "app/shortcuts.cpp")
# add_executable(labels ${SOURCES} "app/labels.cpp")


# target_link_libraries(local LINK_PUBLIC ${LIBS})
# target_link_libraries(shortcuts LINK_PUBLIC ${LIBS})
# target_link_libraries(labels LINK_PUBLIC ${LIBS})
