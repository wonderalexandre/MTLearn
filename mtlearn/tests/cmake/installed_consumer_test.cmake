cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED MTLEARN_BUILD_DIR)
    message(FATAL_ERROR "MTLEARN_BUILD_DIR is required")
endif()

set(work_dir "${MTLEARN_BUILD_DIR}/installed-consumer-test")
set(prefix "${work_dir}/prefix")
set(consumer_source_dir "${work_dir}/consumer")
set(consumer_build_dir "${work_dir}/consumer-build")

file(REMOVE_RECURSE "${work_dir}")
file(MAKE_DIRECTORY "${consumer_source_dir}")

set(install_command
    "${CMAKE_COMMAND}" --install "${MTLEARN_BUILD_DIR}" --prefix "${prefix}")

if(DEFINED MTLEARN_CTEST_CONFIG AND NOT "${MTLEARN_CTEST_CONFIG}" STREQUAL "")
    list(APPEND install_command --config "${MTLEARN_CTEST_CONFIG}")
endif()

execute_process(
    COMMAND ${install_command}
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_output
    ERROR_VARIABLE install_error)

if(NOT install_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to install mtlearn for consumer test.\n"
        "stdout:\n${install_output}\n"
        "stderr:\n${install_error}")
endif()

if(EXISTS "${prefix}/include/mmcfilters")
    message(FATAL_ERROR "Installed mtlearn package must not expose include/mmcfilters")
endif()

if(EXISTS "${prefix}/include/mtlearn/detail")
    message(FATAL_ERROR "Installed mtlearn package must not expose include/mtlearn/detail")
endif()

file(WRITE "${consumer_source_dir}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.20)\n"
"project(mtlearn_installed_consumer LANGUAGES CXX)\n"
"find_package(mtlearn CONFIG REQUIRED)\n"
"if(TARGET mmcfilters::core)\n"
"    message(FATAL_ERROR \"mtlearn consumer should not receive mmcfilters::core\")\n"
"endif()\n"
"add_executable(consumer main.cpp)\n"
"target_link_libraries(consumer PRIVATE mtlearn::core)\n")

file(WRITE "${consumer_source_dir}/main.cpp"
"#include <array>\n"
"#include <cassert>\n"
"#include <cstdint>\n"
"#include <variant>\n"
"#include <mtlearn/morphology.hpp>\n"
"\n"
"int main()\n"
"{\n"
"    namespace morphology = mtlearn::morphology;\n"
"    morphology::AttributeOrGroup request = morphology::AttributeGroup::SHAPE;\n"
"    assert(std::holds_alternative<morphology::AttributeGroup>(request));\n"
"    std::array<std::uint8_t, 4> pixels{1, 2, 3, 4};\n"
"    auto tree = morphology::WeightedTree::createComponentTree({pixels.data(), 2, 2}, true);\n"
"    assert(tree.numRows() == 2);\n"
"    assert(tree.numCols() == 2);\n"
"    assert(tree.numNodes() > 0);\n"
"    const auto reconstructed = tree.reconstructionImage();\n"
"    assert(reconstructed.pixels.size() == pixels.size());\n"
"    auto hierarchy = tree.exportHigraHierarchy();\n"
"    assert(!hierarchy.first.empty());\n"
"    assert(hierarchy.first.size() == hierarchy.second.size());\n"
"    return 0;\n"
"}\n")

set(consumer_configure_command
    "${CMAKE_COMMAND}"
    -S "${consumer_source_dir}"
    -B "${consumer_build_dir}"
    "-DCMAKE_PREFIX_PATH=${prefix}")

if(DEFINED MTLEARN_CTEST_CONFIG AND NOT "${MTLEARN_CTEST_CONFIG}" STREQUAL "")
    list(APPEND consumer_configure_command "-DCMAKE_BUILD_TYPE=${MTLEARN_CTEST_CONFIG}")
endif()

execute_process(
    COMMAND ${consumer_configure_command}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_error)

if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to configure installed mtlearn consumer.\n"
        "stdout:\n${configure_output}\n"
        "stderr:\n${configure_error}")
endif()

set(consumer_build_command "${CMAKE_COMMAND}" --build "${consumer_build_dir}" --parallel)

if(DEFINED MTLEARN_CTEST_CONFIG AND NOT "${MTLEARN_CTEST_CONFIG}" STREQUAL "")
    list(APPEND consumer_build_command --config "${MTLEARN_CTEST_CONFIG}")
endif()

execute_process(
    COMMAND ${consumer_build_command}
    RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_output
    ERROR_VARIABLE build_error)

if(NOT build_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to build installed mtlearn consumer.\n"
        "stdout:\n${build_output}\n"
        "stderr:\n${build_error}")
endif()

set(consumer_executable_name "consumer")
if(WIN32)
    set(consumer_executable_name "consumer.exe")
endif()

set(consumer_executable "${consumer_build_dir}/${consumer_executable_name}")
if(DEFINED MTLEARN_CTEST_CONFIG AND NOT "${MTLEARN_CTEST_CONFIG}" STREQUAL "")
    set(config_consumer_executable "${consumer_build_dir}/${MTLEARN_CTEST_CONFIG}/${consumer_executable_name}")
    if(EXISTS "${config_consumer_executable}")
        set(consumer_executable "${config_consumer_executable}")
    endif()
endif()

execute_process(
    COMMAND "${consumer_executable}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error)

if(NOT run_result EQUAL 0)
    message(FATAL_ERROR
        "Installed mtlearn consumer executable failed.\n"
        "stdout:\n${run_output}\n"
        "stderr:\n${run_error}")
endif()
