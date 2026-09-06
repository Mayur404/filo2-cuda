cmake_minimum_required(VERSION 3.22)
project(filo2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(WARNING_FLAGS "/W3")
    set(OPT_FLAGS "/O2")
else()
    set(WARNING_FLAGS "-Wall -Wextra -Wpedantic -Wuninitialized")
    set(OPT_FLAGS "-O3 -march=native -flto=auto")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address -fsanitize=undefined -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OPT_FLAGS}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

set(CMAKE_VERBOSE_MAKEFILE ON)

option(ENABLE_VERBOSE "Enable verbose output" ON)
option(ENABLE_GUI "Enable graphical interface" OFF)
option(ENABLE_TIMELIMIT "Enable time limiet termination" OFF)
option(FILO2_USE_CUDA "Build and use the optional CUDA k-NN implementation" OFF)

message("-- Build options")

if(ENABLE_VERBOSE)
    message("--- Verbose output ENABLED")
    add_definitions(-DVERBOSE)
else()
    message("--- Verbose output DISABLED")
endif()

if(ENABLE_GUI)
    message("--- Graphical interface ENABLED")
    add_definitions(-DGUI)
    set(LIBRARIES ${LIBRARIES} glfw GL)
    set(SOURCE ${SOURCE} Renderer.hpp)
else()
    message("--- Graphical interface DISABLED")
endif()

if(ENABLE_TIMELIMIT)
    message("--- Time limit termination ENABLED")
    add_definitions(-DTIMELIMIT)
else()
    message("--- Time limit termination DISABLED")
endif()

# CUDA is deliberately opt-in.  Keeping enable_language(CUDA) inside this
# branch means a normal CPU build does not require nvcc, CUDA headers, or a
# CUDA runtime library.  CMake's CUDA compiler detection is performed only
# when the caller explicitly requests FILO2_USE_CUDA=ON.
set(FILO2_CUDA_SOURCES)
set(FILO2_KNN_SOURCES
    knn/GridKnn.cpp
    knn/GridKnn.hpp
)
if(FILO2_USE_CUDA)
    message("--- CUDA k-NN ENABLED")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # Keep the implementation translation unit explicit.  This makes a
    # CUDA-enabled build fail at configure time if the integration source is
    # accidentally omitted from a checkout, rather than silently producing a
    # binary with an unresolved CUDA API.
    set(FILO2_CUDA_SOURCES cuda/CudaGridKnn.cu)
    foreach(FILO2_CUDA_SOURCE IN LISTS FILO2_CUDA_SOURCES)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${FILO2_CUDA_SOURCE}")
            message(FATAL_ERROR
                "FILO2_USE_CUDA=ON but ${FILO2_CUDA_SOURCE} is missing")
        endif()
    endforeach()
    add_compile_definitions(FILO2_USE_CUDA=1)
endif()

add_executable(filo2

    base/BinaryHeap.hpp
    base/BitMatrix.hpp
    base/FixedSizeValueStack.hpp
    base/Flat2DVector.hpp
    base/functor.hpp
    base/KDTree.cpp
    base/KDTree.hpp
    base/LRUCache.hpp
    base/macro.hpp
    base/NonCopyable.hpp
    base/PrettyPrinter.hpp
    base/SmallFlatMap.hpp
    base/SmallFlatSet.hpp
    base/SparseIntSet.hpp
    base/Timer.hpp
    base/VectorView.hpp
    base/Welford.hpp

    instance/Instance.cpp
    instance/Instance.hpp
    instance/Parser.cpp
    instance/Parser.hpp

    localsearch/AbstractOperator.hpp
    localsearch/EjectionChain.hpp
    localsearch/LocalSearch.hpp
    localsearch/OneOneExchange.hpp
    localsearch/OneZeroExchange.hpp
    localsearch/RevThreeOneExchange.hpp
    localsearch/RevThreeThreeExchange.hpp
    localsearch/RevThreeTwoExchange.hpp
    localsearch/RevThreeZeroExchange.hpp
    localsearch/RevTwoOneExchange.hpp
    localsearch/RevTwoTwoExchange.hpp
    localsearch/RevTwoZeroExchange.hpp
    localsearch/SplitExchange.hpp
    localsearch/TailsExchange.hpp
    localsearch/ThreeOneExchange.hpp
    localsearch/ThreeThreeExchange.hpp
    localsearch/ThreeTwoExchange.hpp
    localsearch/ThreeZeroExchange.hpp
    localsearch/TwoOneExchange.hpp
    localsearch/TwoOptExchange.hpp
    localsearch/TwoTwoExchange.hpp
    localsearch/TwoZeroExchange.hpp

    movegen/MoveGenerators.hpp

    opt/routemin.hpp
    opt/RuinAndRecreate.hpp
    opt/SimulatedAnnealing.hpp

    solution/savings.hpp
    solution/Solution.hpp
    solution/Solution.cpp

    main.cpp
    Parameters.hpp
    Renderer.hpp

    ${FILO2_KNN_SOURCES}
    ${FILO2_CUDA_SOURCES}
)

target_link_libraries(filo2 PUBLIC ${LIBRARIES})

if(FILO2_USE_CUDA)
    # The host code calls the CUDA API, while the .cu implementation owns
    # device code.  Link cudart only in the CUDA configuration.
    target_link_libraries(filo2 PUBLIC CUDA::cudart)
    set_target_properties(filo2 PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# CPU and host-only planner tests remain available in the default build.  The
# runtime GPU-vs-CPU test is configured only when FILO2_USE_CUDA explicitly
# signals that a CUDA toolkit is available.
include(CTest)
if(BUILD_TESTING)
    add_executable(cpu_grid_knn_test
        tests/cpu_grid_knn_test.cpp
        ${FILO2_KNN_SOURCES}
    )
    target_compile_features(cpu_grid_knn_test PRIVATE cxx_std_17)
    target_include_directories(cpu_grid_knn_test
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME cpu_grid_knn_test COMMAND cpu_grid_knn_test)

    # The allocation planner is deliberately host-only.  Keep this test in
    # the default build so its overflow and boundary checks run even on a
    # machine with neither a CUDA toolkit nor a GPU.
    add_executable(cuda_memory_plan_test
        tests/cuda_memory_plan_test.cpp
    )
    target_compile_features(cuda_memory_plan_test PRIVATE cxx_std_17)
    target_include_directories(cuda_memory_plan_test
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME cuda_memory_plan_test COMMAND cuda_memory_plan_test)

    # Direct head-to-head test: base/KDTree.cpp vs knn/GridKnn.cpp
    add_executable(kdtree_vs_gridknn_test
        tests/kdtree_vs_gridknn_test.cpp
        base/KDTree.cpp
        base/KDTree.hpp
        ${FILO2_KNN_SOURCES}
    )
    target_compile_features(kdtree_vs_gridknn_test PRIVATE cxx_std_17)
    target_include_directories(kdtree_vs_gridknn_test
        PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME kdtree_vs_gridknn_test COMMAND kdtree_vs_gridknn_test)
endif()

if(BUILD_TESTING AND FILO2_USE_CUDA)
    set(FILO2_CUDA_TEST_SOURCES
        tests/cuda_grid_knn_test.cu
    )
    foreach(FILO2_CUDA_TEST_SOURCE IN LISTS FILO2_CUDA_TEST_SOURCES)
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${FILO2_CUDA_TEST_SOURCE}")
            message(FATAL_ERROR
                "FILO2_USE_CUDA=ON but test source ${FILO2_CUDA_TEST_SOURCE} is missing")
        endif()
        get_filename_component(FILO2_CUDA_TEST_NAME
            "${FILO2_CUDA_TEST_SOURCE}" NAME_WE)
        add_executable(${FILO2_CUDA_TEST_NAME}
            "${FILO2_CUDA_TEST_SOURCE}"
            ${FILO2_KNN_SOURCES}
            ${FILO2_CUDA_SOURCES}
        )
        target_compile_features(${FILO2_CUDA_TEST_NAME} PRIVATE cxx_std_17)
        target_include_directories(${FILO2_CUDA_TEST_NAME}
            PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(${FILO2_CUDA_TEST_NAME} PRIVATE CUDA::cudart)
        set_target_properties(${FILO2_CUDA_TEST_NAME} PROPERTIES
            CUDA_STANDARD 17
            CUDA_STANDARD_REQUIRED ON
            CUDA_EXTENSIONS OFF
            CUDA_SEPARABLE_COMPILATION ON
        )
        add_test(NAME ${FILO2_CUDA_TEST_NAME}
                 COMMAND ${FILO2_CUDA_TEST_NAME})
        # A CUDA-enabled build can still be configured on a machine where no
        # device is visible.  Return code 77 means the runtime-only test is
        # skipped rather than misreported as a correctness failure.
        set_tests_properties(${FILO2_CUDA_TEST_NAME} PROPERTIES
            SKIP_RETURN_CODE 77)
    endforeach()
endif()
