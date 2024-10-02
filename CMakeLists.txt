cmake_minimum_required(VERSION 3.16)

PROJECT(supr
    VERSION 0.1
    DESCRIPTION "SUPR in cpp"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------
# cmake options

option(SUPR_BUILD_EXAMPLES "" OFF)
option(BUILD_WITH_CUDA "" ON)

include(cmake/helper_macros.cmake)

include(FetchContent)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)


if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # if you want to have .profraw files use this flag: -fprofile-generate
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v -fvectorize")

    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_86")
endif()

set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O3 -Wno-unused-parameter -Wno-unused-variable -Wno-int-to-pointer-cast -march=native -funroll-loops -finline-functions")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-declarations -pthread")

# linker flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")

# position independent code is required for shared libs. For GCC this is -fPIC
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)



option(CAFFE2_USE_CUDNN "Whether torch should use cudnn" ON)
##Torch
#execute_process(COMMAND python -c "import torch;print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE TORCHPREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
#list(APPEND CMAKE_PREFIX_PATH "${TORCHPREFIX}")
find_package(Torch REQUIRED)
#LibraryHelper(TORCH_LIBRARIES libnvToolsExt.so "${CUDA_TOOLKIT_ROOT_DIR}/lib" Torch_FOUND)
#list(REMOVE_ITEM TORCH_LIBRARIES LIBNVTOOLSEXT-NOTFOUND)
#message(STATUS "Libstorch: ${TORCH_LIBRARIES}")
#PackageHelper(Torch "${Torch_FOUND}" "${TORCH_INCLUDE_DIRS}" "${TORCH_LIBRARIES}")
PackageHelperTarget(torch TORCH_FOUND)
#message(STATUS "TORCH_FLAGS: ${TORCH_CXX_FLAGS}, LIBS: ${TORCH_LIBRARIES}, INCLUDES: ${TORCH_INCLUDES}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -frecord-gcc-switches")
set(LIBS ${LIBS} ${TORCH_LIBRARIES})

# # -----------------------------------------------------
# # -----------------------------------------------------

# # CUDA
if (BUILD_WITH_CUDA)
    find_package(CUDA 11.8 REQUIRED)

    #(CUDA_LIBRARIES ${CUDA_LIBRARIES} cuda cublas)
    set(CUDA_LIBRARIES ${CUDA_LIBRARIES} cuda)
    PackageHelper(CUDA "${CUDA_FOUND}" "${CUDA_INCLUDE_DIRS}" "${CUDA_LIBRARIES}")

    OPTION(CUDA_PROFILING "Enables the PROFILING #define" OFF)
    OPTION(CUDA_DEBUG "Enables CUDA Debugging in Nsight" OFF)
    OPTION(FULL_SPEED "Enables All Optimizations" OFF)


    if (CUDA_PROFILING)
        add_definitions(-DCUDA_PROFILING)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
        SET(CMAKE_BUILD_TYPE RelWithDebInfo)
    endif ()

    if (CUDA_DEBUG)
        SET(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
        SET(CMAKE_BUILD_TYPE DEB1G)
        add_definitions(-DCUDA_DEBUG)
    else ()
        add_definitions(-DCUDA_NDEBUG)
    endif ()


    if (NOT MSVC)
        list(APPEND CMAKE_CUDA_FLAGS "-Xcompiler=-fopenmp")
        list(APPEND CMAKE_CUDA_FLAGS "-Xcompiler=-march=native")
    endif ()

    list(APPEND CMAKE_CUDA_FLAGS "-Xcudafe=--diag_suppress=field_without_dll_interface")
    list(APPEND CMAKE_CUDA_FLAGS "-Xcudafe=--diag_suppress=base_class_has_different_dll_interface")
    list(APPEND CMAKE_CUDA_FLAGS "-Xcudafe=--diag_suppress=dll_interface_conflict_none_assumed")
    list(APPEND CMAKE_CUDA_FLAGS "-Xcudafe=--diag_suppress=dll_interface_conflict_dllexport_assumed")


    list(APPEND CMAKE_CUDA_FLAGS "-use_fast_math")
    list(APPEND CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")
    list(APPEND CMAKE_CUDA_FLAGS "-Xcudafe=--diag_suppress=esa_on_defaulted_function_ignored")

endif()



# cnpy
fetchcontent_declare(
        cnpy
        GIT_REPOSITORY "https://github.com/rogersce/cnpy.git")
FetchContent_MakeAvailable(cnpy)

PackageHelper(cnpy TRUE "${cnpy_SOURCE_DIR}" "${cnpy_LIBRARIES}")

FetchContent_Declare(
        json
        GIT_REPOSITORY "https://github.com/nlohmann/json.git")
FetchContent_MakeAvailable(json)
IncludeHelper(json_INCLUDE_DIRS nlohmann/json.hpp "${json_SOURCE_DIR}/single_include/" json_FOUND)
PackageHelper(json "${json_FOUND}" "${json_INCLUDE_DIRS}" "")


add_library(supr SHARED "${PROJECT_SOURCE_DIR}/suprcpp.cpp" "${PROJECT_SOURCE_DIR}/suprcpp.h" "${PROJECT_SOURCE_DIR}/suprutil.cpp" "${PROJECT_SOURCE_DIR}/suprutil.h")

include(GenerateExportHeader)
generate_export_header(supr)

target_include_directories(supr PUBLIC .)

target_include_directories(supr PUBLIC ${PACKAGE_INCLUDES})

target_link_libraries(supr PUBLIC ${LIBS} ${LIB_TARGETS} cnpy)


if (SUPR_BUILD_EXAMPLES)
    add_subdirectory(example)
endif()