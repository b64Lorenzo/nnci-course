cmake_minimum_required(VERSION 3.20)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" NNCI_VERSION)
string(STRIP "${NNCI_VERSION}" NNCI_VERSION)
if(NOT NNCI_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "VERSION must contain a semantic version in MAJOR.MINOR.PATCH format")
endif()

project(nnci_course VERSION "${NNCI_VERSION}" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(nnci_logging src/logging.cpp)
target_include_directories(nnci_logging PUBLIC include)

add_library(nnci_perceptron
    src/assignment1_study.cpp
    src/rosenblatt_perceptron.cpp
)
target_include_directories(nnci_perceptron PUBLIC include)
target_link_libraries(nnci_perceptron PUBLIC nnci_logging)

add_executable(nnci src/main.cpp)
target_link_libraries(nnci PRIVATE nnci_logging nnci_perceptron)

set(NNCI_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${NNCI_GENERATED_INCLUDE_DIR}/nnci")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in"
    "${NNCI_GENERATED_INCLUDE_DIR}/nnci/version.hpp"
    @ONLY
)
target_include_directories(nnci PRIVATE "${NNCI_GENERATED_INCLUDE_DIR}")

if(WIN32)
    enable_language(RC)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/resources/version.rc.in"
        "${CMAKE_CURRENT_BINARY_DIR}/version.rc"
        @ONLY
    )
    target_sources(nnci PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/version.rc")
endif()

if(MSVC)
    target_compile_options(nnci_logging PRIVATE /W4 /permissive-)
    target_compile_options(nnci_perceptron PRIVATE /W4 /permissive-)
    target_compile_options(nnci PRIVATE /W4 /permissive-)
else()
    target_compile_options(nnci_logging PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(nnci_perceptron PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(nnci PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(nnci_perceptron_tests tests/assignment_tests.cpp)
    target_link_libraries(nnci_perceptron_tests PRIVATE nnci_perceptron)

    if(MSVC)
        target_compile_options(nnci_perceptron_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(nnci_perceptron_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    add_test(NAME nnci_cli_version COMMAND nnci --version)
    set_tests_properties(nnci_cli_version PROPERTIES
        PASS_REGULAR_EXPRESSION "nnci ${PROJECT_VERSION}"
    )
    if(WIN32)
        find_program(POWERSHELL_EXECUTABLE NAMES pwsh powershell REQUIRED)
        add_test(NAME nnci_version_metadata
            COMMAND "${POWERSHELL_EXECUTABLE}" -NoProfile -ExecutionPolicy Bypass
                    -File "${CMAKE_SOURCE_DIR}/tests/verify_exe_version.ps1"
                    -Executable "$<TARGET_FILE:nnci>"
                    -ExpectedVersion "${PROJECT_VERSION}"
        )
    endif()

    add_test(NAME a1_study_smoke COMMAND nnci a1-study
        --dimensions 4 --datasets 2 --max-epochs 5
        --p-min 1 --p-max 4 --p-step 1 --seed 13
        --output a1_capacity_test.csv
    )
    set_tests_properties(a1_study_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "Wrote 4 capacity points"
    )

    add_test(NAME a1_study_options_log COMMAND nnci a1-study
        --dimensions 4 --datasets 2 --max-epochs 5
        --p-min 1 --p-max 4 --p-step 1 --seed 13
        --output a1_options_capacity.csv
    )
    set_tests_properties(a1_study_options_log PROPERTIES
        PASS_REGULAR_EXPRESSION "options used: dimensions=4, datasets=2, max_epochs=5, p_min=1, p_max=4, p_step=1, seed=13, output=a1_options_capacity.csv"
    )

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME plot_capacity_unit
            COMMAND "${Python3_EXECUTABLE}" -m unittest discover
                    -s "${CMAKE_SOURCE_DIR}/tests" -p test_plot_capacity.py
        )
        set_tests_properties(plot_capacity_unit PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    endif()
endif()
