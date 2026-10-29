cmake_minimum_required(VERSION 3.14)

# ---------------------------------------------------------------------------
# mtlearn test configuration.
# ---------------------------------------------------------------------------
# This file creates C++ test executables, Python test commands, and the
# dependencies required for binding-related test cases.

set(_mtlearn_build_python_tests OFF)
if(MTLEARN_BUILD_PYTHON AND MTLEARN_WITH_TORCH)
    set(_mtlearn_build_python_tests ON)
endif()

function(mtl_add_test TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    target_link_libraries(${TARGET_NAME} PRIVATE mtlearn::core)
    target_compile_features(${TARGET_NAME} PRIVATE cxx_std_20)
    target_compile_definitions(${TARGET_NAME} PRIVATE
        MTLEARN_PYTHON_DIR="${PROJECT_SOURCE_DIR}/mtlearn/python"
        MTLEARN_BINDINGS_DIR="${CMAKE_BINARY_DIR}/mtlearn/bindings"
        MTLEARN_EMBED_DEFAULT=$<BOOL:${MTLEARN_ENABLE_EMBED}>
        MTLEARN_WITH_TORCH=$<BOOL:${MTLEARN_WITH_TORCH}>)
    if(Python3_INCLUDE_DIRS)
        target_include_directories(${TARGET_NAME} PRIVATE ${Python3_INCLUDE_DIRS})
    endif()
    if(TARGET Python3::Embed)
        target_link_libraries(${TARGET_NAME} PRIVATE Python3::Embed)
    elseif(TARGET Python3::Module)
        target_link_libraries(${TARGET_NAME} PRIVATE Python3::Module)
    endif()
    if(TARGET pybind11::embed)
        target_link_libraries(${TARGET_NAME} PRIVATE pybind11::embed)
    endif()
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
endfunction()

if(_mtlearn_build_python_tests)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED
        OPTIONAL_COMPONENTS Development.Embed)
endif()

if(_mtlearn_build_python_tests)
    if(NOT TARGET pybind11::embed)
        find_package(pybind11 CONFIG QUIET)
    endif()

    if(NOT TARGET pybind11::embed)
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.1.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# Try to locate a Python interpreter with torch for test execution.
if(_mtlearn_build_python_tests)
    set(_mtlearn_python_exec "${Python3_EXECUTABLE}")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "-c" "import torch"
        RESULT_VARIABLE _mtlearn_python_has_torch
        OUTPUT_QUIET
        ERROR_QUIET
    )

    if(NOT _mtlearn_python_has_torch EQUAL 0)
        if(DEFINED ENV{CONDA_PREFIX})
            foreach(_suffix python3.12 python3 python)
                set(_candidate "$ENV{CONDA_PREFIX}/bin/${_suffix}")
                if(EXISTS "${_candidate}")
                    set(_mtlearn_python_exec "${_candidate}")
                    break()
                endif()
            endforeach()
        endif()
    endif()

    if(NOT _mtlearn_python_has_torch EQUAL 0)
        if(NOT EXISTS "${_mtlearn_python_exec}")
            set(_mtlearn_python_exec "/opt/anaconda3/bin/python3.12")
        endif()
    endif()

    if(NOT "${_mtlearn_python_exec}" STREQUAL "${Python3_EXECUTABLE}" AND EXISTS "${_mtlearn_python_exec}")
        set(Python3_EXECUTABLE "${_mtlearn_python_exec}" CACHE FILEPATH "Python interpreter with torch" FORCE)
    endif()
endif()

if(_mtlearn_build_python_tests)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "-m" "pytest" "--version"
        RESULT_VARIABLE _mtlearn_python_has_pytest
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT _mtlearn_python_has_pytest EQUAL 0)
        message(FATAL_ERROR
            "Python tests require pytest. Install test dependencies with "
            "`pip install -e .[test]`.")
    endif()

    set(_mtlearn_path_separator ":")
    if(WIN32)
        set(_mtlearn_path_separator "$<SEMICOLON>")
    endif()
    set(_mtlearn_pythonpath "${PROJECT_SOURCE_DIR}/mtlearn/python${_mtlearn_path_separator}${CMAKE_BINARY_DIR}/mtlearn/bindings")

    add_test(
        NAME mtl_python_bindings
        COMMAND
            ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${_mtlearn_pythonpath}"
                ${Python3_EXECUTABLE} -m pytest -q -m "not gradcheck" ${CMAKE_CURRENT_SOURCE_DIR}/python
    )
    add_test(
        NAME mtl_python_gradchecks
        COMMAND
            ${CMAKE_COMMAND} -E env
                "PYTHONPATH=${_mtlearn_pythonpath}"
                ${Python3_EXECUTABLE} -m pytest -q -m gradcheck ${CMAKE_CURRENT_SOURCE_DIR}/python
    )
endif()

# C++ tests for the public facade and, when the Python extension is built, the
# optional embedded interpreter test.
mtl_add_test(mtl_public_morphology_test cpp/public_morphology.cpp)

add_test(
    NAME mtl_installed_consumer_test
    COMMAND
        ${CMAKE_COMMAND}
            -DMTLEARN_BUILD_DIR=${CMAKE_BINARY_DIR}
            -DMTLEARN_CTEST_CONFIG=$<CONFIG>
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/installed_consumer_test.cmake)

if(_mtlearn_build_python_tests)
    mtl_add_test(mtl_interpreter_test cpp/test_interpreter.cpp)
endif()

# In the `ctest` flow, C++ tests inherit PYTHONPATH and keep embed mode off.
# For local debugging, export MTLEARN_ENABLE_EMBED=1 before running
# `mtl_interpreter_test`.
if(MTLEARN_ENABLE_EMBED)
    set(_mtlearn_embed_env "MTLEARN_ENABLE_EMBED=1")
else()
    set(_mtlearn_embed_env "MTLEARN_ENABLE_EMBED=0")
endif()

set(_mtlearn_test_env "MTLEARN_ENABLE_EMBED=0")
if(_mtlearn_build_python_tests)
    set(_mtlearn_test_env "PYTHONPATH=${_mtlearn_pythonpath};${_mtlearn_embed_env}")
endif()
if(_mtlearn_build_python_tests)
    set_tests_properties(mtl_interpreter_test PROPERTIES ENVIRONMENT "${_mtlearn_test_env}")
endif()
if(_mtlearn_build_python_tests)
    set_tests_properties(mtl_python_bindings PROPERTIES ENVIRONMENT "${_mtlearn_test_env}")
endif()
if(_mtlearn_build_python_tests)
    set_tests_properties(mtl_python_gradchecks PROPERTIES ENVIRONMENT "${_mtlearn_test_env}")
endif()
set_tests_properties(mtl_public_morphology_test PROPERTIES ENVIRONMENT "${_mtlearn_test_env}")
