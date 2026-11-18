cmake_minimum_required(VERSION 3.20)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" NNCI_VERSION)
string(STRIP "${NNCI_VERSION}" NNCI_VERSION)
if(NOT NNCI_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    message(FATAL_ERROR "VERSION must contain a semantic version in MAJOR.MINOR.PATCH format")
endif()

project(nnci_course VERSION "${NNCI_VERSION}" LANGUAGES CXX)

option(NNCI_ENABLE_CUDA "Build the opt-in CUDA perceptron backend" OFF)
if(NNCI_ENABLE_CUDA)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
        set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures (real code and PTX)")
    endif()
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

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
if(NNCI_ENABLE_CUDA)
    target_sources(nnci_perceptron PRIVATE src/perceptron_cuda.cu)
    set_property(TARGET nnci_perceptron PROPERTY CUDA_RUNTIME_LIBRARY Static)
    target_link_libraries(nnci_perceptron PRIVATE CUDA::cudart_static)
    target_compile_options(nnci_perceptron PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
else()
    target_sources(nnci_perceptron PRIVATE src/perceptron_cuda_stub.cpp)
endif()

add_executable(nnci
    src/main.cpp
    src/app/cli.cpp
    src/app/perceptron_options.cpp
    src/app/perceptron_command.cpp
    src/app/study_output.cpp
    src/app/plot_runner.cpp
)
target_link_libraries(nnci PRIVATE nnci_logging nnci_perceptron)
add_custom_command(TARGET nnci POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:nnci>/scripts"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_capacity.py"
            "$<TARGET_FILE_DIR:nnci>/scripts/plot_capacity.py"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_strength.py"
            "$<TARGET_FILE_DIR:nnci>/scripts/plot_strength.py"
    COMMENT "Copying plotting scripts beside nnci"
)

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
    target_compile_options(nnci_perceptron PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive->")
    target_compile_options(nnci PRIVATE /W4 /permissive-)
else()
    target_compile_options(nnci_logging PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(nnci_perceptron PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>")
    target_compile_options(nnci PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    if(NNCI_ENABLE_CUDA)
        add_executable(nnci_cuda_tests tests/cuda_tests.cpp)
        target_link_libraries(nnci_cuda_tests PRIVATE nnci_perceptron)
        add_test(NAME nnci_cuda_differential COMMAND nnci_cuda_tests)
        set_tests_properties(nnci_cuda_differential PROPERTIES LABELS "cuda;gpu" TIMEOUT 90)
    else()
        add_executable(nnci_cuda_unavailable_tests tests/cuda_unavailable_tests.cpp)
        target_link_libraries(nnci_cuda_unavailable_tests PRIVATE nnci_perceptron)
        add_test(NAME nnci_cuda_unavailable COMMAND nnci_cuda_unavailable_tests)
    endif()

    add_executable(nnci_perceptron_tests tests/assignment_tests.cpp)
    target_link_libraries(nnci_perceptron_tests PRIVATE nnci_perceptron)

    add_executable(nnci_cli_options_tests
        tests/cli_options_tests.cpp
        src/app/perceptron_options.cpp
    )
    target_include_directories(nnci_cli_options_tests PRIVATE src/app)

    if(MSVC)
        target_compile_options(nnci_perceptron_tests PRIVATE /W4 /permissive-)
        target_compile_options(nnci_cli_options_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(nnci_perceptron_tests PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(nnci_cli_options_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()
    add_test(NAME nnci_perceptron_unit COMMAND nnci_perceptron_tests)
    add_test(NAME nnci_cli_options COMMAND nnci_cli_options_tests)
    add_test(NAME perceptron_theoretical_validation
        COMMAND nnci perceptron --validation
    )
    set_tests_properties(perceptron_theoretical_validation PROPERTIES
        PASS_REGULAR_EXPRESSION "PASS theoretical validation"
    )

    add_test(NAME nnci_cli_version COMMAND nnci --version)
    set_tests_properties(nnci_cli_version PROPERTIES
        PASS_REGULAR_EXPRESSION "nnci ${PROJECT_VERSION}"
    )
    add_test(NAME nnci_cli_help COMMAND nnci --help)
    set_tests_properties(nnci_cli_help PROPERTIES
        PASS_REGULAR_EXPRESSION "--validation"
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

    add_test(NAME perceptron_capacity_smoke
        COMMAND nnci perceptron --capacity
            --dimensions 4
            --datasets 2
            --max-epochs 5
            --p-min 1
            --p-max 4
            --p-step 1
            --seed 13
            --threshold 0
            --output results/capacity_smoke.csv
    )
    set_tests_properties(perceptron_capacity_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "Wrote 4 capacity points"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-working-directory"
    )


    add_test(NAME perceptron_ncapacity_combined_plot_smoke
        COMMAND nnci perceptron --NCapacity 2 4 2
            --datasets 2
            --max-epochs 5
            --seed 13
    )
    set_tests_properties(perceptron_ncapacity_combined_plot_smoke PROPERTIES
        PASS_REGULAR_EXPRESSION "Saved combined capacity plot to"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-working-directory"
    )

    add_test(NAME perceptron_strength_plot_external_working_directory
        COMMAND nnci perceptron --strength
            --dimensions 2
            --patterns 2
            --datasets 10
            --max-epochs 100
            --seed 4
    )
    set(NNCI_TEST_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-working-directory")
    file(MAKE_DIRECTORY "${NNCI_TEST_WORKING_DIRECTORY}")
    set_tests_properties(perceptron_strength_plot_external_working_directory PROPERTIES
        PASS_REGULAR_EXPRESSION "Saved strength histogram to"
        WORKING_DIRECTORY "${NNCI_TEST_WORKING_DIRECTORY}"
    )

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME perceptron_cuda_cli
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_cuda_cli.py"
                    "$<TARGET_FILE:nnci>" "$<IF:$<BOOL:${NNCI_ENABLE_CUDA}>,on,off>"
        )
        set_tests_properties(perceptron_cuda_cli PROPERTIES TIMEOUT 180 LABELS cuda)
        if(NNCI_ENABLE_CUDA)
            set_tests_properties(perceptron_cuda_cli PROPERTIES LABELS "cuda;gpu")
        endif()
        add_test(NAME plot_capacity_unit
            COMMAND "${Python3_EXECUTABLE}" -m unittest discover
                    -s "${CMAKE_SOURCE_DIR}/tests" -p test_plot_capacity.py
        )
        set_tests_properties(plot_capacity_unit PROPERTIES
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
        add_test(NAME perceptron_ncapacity_custom_pattern_range
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/tests/test_ncapacity_cli.py"
                    "$<TARGET_FILE:nnci>"
        )
    endif()
endif()

