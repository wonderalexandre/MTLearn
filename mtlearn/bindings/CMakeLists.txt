# ---------------------------------------------------------------------------
# pybind11 module configuration (_mtlearn)
# ---------------------------------------------------------------------------

find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        URL https://github.com/pybind/pybind11/archive/refs/tags/v3.0.1.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)

if(MTLEARN_WITH_TORCH)
    find_package(Torch REQUIRED)
    pybind11_add_module(_mtlearn MtlearnModule.cpp)
    # CMake may report optional Torch libraries as NOTFOUND on some platforms.
    # Keep the link interface explicit and platform-neutral so Linux, macOS,
    # and Windows wheels all link against the same project and Torch targets.
    set(_mtlearn_torch_libraries "")
    foreach(lib ${TORCH_LIBRARIES})
        if(NOT lib MATCHES "-NOTFOUND$")
            list(APPEND _mtlearn_torch_libraries ${lib})
        endif()
    endforeach()
    target_link_libraries(_mtlearn PRIVATE mtlearn::core mmcfilters::core ${_mtlearn_torch_libraries})

    if(DEFINED TORCH_LIBRARY AND TORCH_LIBRARY)
        get_filename_component(_torch_root "${TORCH_LIBRARY}" DIRECTORY)
        find_library(TORCH_PYTHON_LIBRARY torch_python
            HINTS "${_torch_root}" "${_torch_root}/.." "${_torch_root}/../.."
            PATH_SUFFIXES lib)
        if(TORCH_PYTHON_LIBRARY)
            target_link_libraries(_mtlearn PRIVATE "${TORCH_PYTHON_LIBRARY}")
        endif()
    endif()
    target_include_directories(_mtlearn BEFORE PRIVATE ${pybind11_INCLUDE_DIRS})
    target_include_directories(_mtlearn PRIVATE
        ${PROJECT_SOURCE_DIR}/mtlearn/include
        ${PROJECT_SOURCE_DIR}/mtlearn/src
        ${CMAKE_SOURCE_DIR}/external/mmcfilters
        ${TORCH_INCLUDE_DIRS})
    target_compile_definitions(_mtlearn PRIVATE MTLEARN_WITH_TORCH=1)
endif()

set_target_properties(_mtlearn PROPERTIES
    OUTPUT_NAME "_mtlearn"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
foreach(_mtlearn_config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set_target_properties(_mtlearn PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${_mtlearn_config} "${CMAKE_CURRENT_BINARY_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_${_mtlearn_config} "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()
set_property(TARGET _mtlearn PROPERTY CXX_STANDARD 20)
set_property(TARGET _mtlearn PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET _mtlearn PROPERTY CXX_EXTENSIONS OFF)
set_target_properties(_mtlearn PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS _mtlearn
        LIBRARY DESTINATION mtlearn
        ARCHIVE DESTINATION mtlearn
        RUNTIME DESTINATION mtlearn)
