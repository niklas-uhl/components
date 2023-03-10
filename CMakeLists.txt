cmake_minimum_required(VERSION 3.15)
project(components LANGUAGES CXX)

find_package(MPI REQUIRED)

#set(KACOMP_DEFINITIONS "")
option(KACOMP_TIEBREAK_DEGREE "Tiebreak based on degree instead of ID." OFF)
option(KACOMP_MEMINFO "Show memory consumed." OFF)
option(KACOMP_USE_CONNECT_IO "Use connect IO for reading files" ON)

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

include(FetchContent)
FetchContent_Declare(bakward-mpi
    GIT_REPOSITORY https://github.com/kamping-site/bakward-mpi
    GIT_TAG master
)
FetchContent_MakeAvailable(bakward-mpi)

add_library(KaCC ${KACOMP_SOURCES})
target_include_directories(KaCC PUBLIC "${KACOMP_INCLUDE_DIRS}")
target_compile_definitions(KaCC PUBLIC ${KACOMP_DEFINITIONS})

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/extern/ips4o" EXCLUDE_FROM_ALL)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/extern/kagen" EXCLUDE_FROM_ALL)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Foo")
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/extern/bliss")
add_library(bliss INTERFACE)
target_include_directories(bliss INTERFACE "${CMAKE_CURRENT_LIST_DIR}/extern/bliss/retired/src/")   #BLISS cmake needs to be executed
target_include_directories(bliss INTERFACE "${CMAKE_CURRENT_LIST_DIR}/extern/bliss/src")
target_include_directories(bliss INTERFACE "${CMAKE_CURRENT_LIST_DIR}/extern/bliss/ext/mxx/include/")   #BLISS cmake needs to be executed
target_include_directories(bliss INTERFACE "${CMAKE_CURRENT_BINARY_DIR}")
set(LIBS
        spooky
        sampling
        MPI::MPI_CXX
        KaGen::KaGen
        ips4o
        bliss
        #kahip
        # asan
        )

target_link_libraries(KaCC PUBLIC ${LIBS})
target_link_libraries(KaCC PUBLIC backward_object backward)

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
