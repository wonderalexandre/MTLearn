cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# Main mtlearn project
# ---------------------------------------------------------------------------
# This file configures the CMake build for the C++ library, Python bindings,
# and test executables. It assumes mmcfilters is available as a submodule at
# external/mmcfilters.

project(mtlearn LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Options exposed to CMake and pyproject.toml consumers.
# - MTLEARN_BUILD_PYTHON : builds the pybind11 module exported to Python.
# - MTLEARN_BUILD_TESTS  : builds C++ tests and helper scripts.
# - MTLEARN_ENABLE_ASSERTS: controls runtime assertions in the core library.
# - MTLEARN_ENABLE_EMBED : enables tests that initialize an embedded Python
#   interpreter. This is useful only with a complete PyTorch environment.
option(MTLEARN_BUILD_PYTHON "Build pybind11 bindings" ON)
option(MTLEARN_BUILD_TESTS "Build C++ tests" OFF)
option(MTLEARN_ENABLE_ASSERTS "Keep runtime asserts even in Release" OFF)
option(MTLEARN_ENABLE_EMBED "Enable embedded Python interpreter tests" OFF)
option(MTLEARN_WITH_TORCH "Build bindings/tests that depend on LibTorch" ON)

if(MTLEARN_BUILD_PYTHON AND NOT MTLEARN_WITH_TORCH)
    message(FATAL_ERROR
        "MTLEARN_BUILD_PYTHON requires MTLEARN_WITH_TORCH because the current "
        "pybind module exposes torch tensors. Disable MTLEARN_BUILD_PYTHON for "
        "a C++-only build.")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Extra directory for project-specific CMake helpers.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Honor variables provided by the command line or pyproject.toml so all
# subprojects use the same Python interpreter.
if(DEFINED PYTHON_EXECUTABLE AND NOT DEFINED Python3_EXECUTABLE)
    set(Python3_EXECUTABLE "${PYTHON_EXECUTABLE}" CACHE FILEPATH "Python interpreter" FORCE)
endif()
if(DEFINED PYTHON_LIBRARY_DIR AND NOT DEFINED Python3_LIBRARY_DIR)
    set(Python3_LIBRARY_DIR "${PYTHON_LIBRARY_DIR}" CACHE PATH "Python library directory" FORCE)
endif()
if(DEFINED Python3_EXECUTABLE AND NOT DEFINED Python3_ROOT_DIR)
    get_filename_component(_mtlearn_py_bin "${Python3_EXECUTABLE}" DIRECTORY)
    get_filename_component(_mtlearn_py_root "${_mtlearn_py_bin}" DIRECTORY)
    set(Python3_ROOT_DIR "${_mtlearn_py_root}" CACHE PATH "Python root" FORCE)
endif()

set(MTLEARN_MMCFILTERS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/mmcfilters")
# Ensure the required mmcfilters submodule is available.
if(NOT EXISTS "${MTLEARN_MMCFILTERS_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "mmcfilters dependency not found at ${MTLEARN_MMCFILTERS_DIR}. Initialize the submodule or adjust the path.")
endif()

# mtlearn owns the Python morphology facade through _mtlearn. The vendored
# backend is consumed only as C++ headers/targets here, so do not build or
# install the top-level mmcfilters Python package from this project.
set(MMCFILTERS_BUILD_PYTHON OFF CACHE BOOL "Build the vendored mmcfilters pybind11 module" FORCE)

add_subdirectory(${MTLEARN_MMCFILTERS_DIR} mmcfilters EXCLUDE_FROM_ALL)

set(_mmc_target "")
if(TARGET mmcfilters_lib)
    set(_mmc_target mmcfilters_lib)
elseif(TARGET mmcfilters::core)
    set(_mmc_target mmcfilters::core)
endif()

if(NOT _mmc_target)
    message(FATAL_ERROR "Could not locate the main mmcfilters target.")
endif()

# mmcfilters provides the core morphology algorithms. Adjust assertion flags
# according to the option selected for this project.
if(MTLEARN_ENABLE_ASSERTS)
    target_compile_definitions(${_mmc_target} INTERFACE MMCFILTERS_ENABLE_ASSERTS)
    target_compile_options(${_mmc_target}
        INTERFACE
            $<$<CXX_COMPILER_ID:MSVC>:/UNDEBUG>
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-UNDEBUG>
    )
else()
    target_compile_definitions(${_mmc_target} INTERFACE NDEBUG)
endif()

# Directory with mtlearn's own C++ libraries and applications.
add_subdirectory(mtlearn/src)

if(MTLEARN_BUILD_PYTHON)
    # Include the pybind11 bindings that produce the _mtlearn module and the
    # corresponding Python package.
    add_subdirectory(mtlearn/bindings)
endif()

if(MTLEARN_BUILD_TESTS)
    enable_testing()
    # C++ test suite, including smoke tests and the optional embedded Python
    # interpreter test.
    add_subdirectory(mtlearn/tests)
endif()

install(EXPORT mtlearnTargets
        NAMESPACE mtlearn::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mtlearn)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/mtlearnConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/mtlearnConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mtlearn)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/mtlearnConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mtlearn)
