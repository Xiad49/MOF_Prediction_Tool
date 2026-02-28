cmake_minimum_required(VERSION 3.16)

project(MOF_Prediction_Tool
    VERSION 0.1.0
    DESCRIPTION "MOF Property Prediction Tool (C++ ML pipeline)"
    LANGUAGES CXX
)

# ------------------------------------------------------------
# Options
# ------------------------------------------------------------
option(MOF_BUILD_TESTS "Build unit tests from tests/*.cpp" OFF)
option(MOF_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# ------------------------------------------------------------
# C++ standard (editable from command line if needed)
# Example: -DCMAKE_CXX_STANDARD=20
# ------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to use (17 or 20)")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------
# Helper: warnings
# ------------------------------------------------------------
function(mof_enable_warnings target_name)
    if (MSVC)
        target_compile_options(${target_name} PRIVATE /W4 /permissive-)
        if (MOF_WARNINGS_AS_ERRORS)
            target_compile_options(${target_name} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target_name} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
        )
        if (MOF_WARNINGS_AS_ERRORS)
            target_compile_options(${target_name} PRIVATE -Werror)
        endif()
    endif()
endfunction()

# ------------------------------------------------------------
# Source discovery (auto-picks new files added under src/)
# "CONFIGURE_DEPENDS" lets CMake re-scan when files are added.
# ------------------------------------------------------------
file(GLOB MOF_ALL_SRC CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

set(MOF_MAIN_SRC "")
set(MOF_CORE_SRC "")

foreach(src_file IN LISTS MOF_ALL_SRC)
    get_filename_component(src_name "${src_file}" NAME)
    if (src_name STREQUAL "main.cpp")
        set(MOF_MAIN_SRC "${src_file}")
    else()
        list(APPEND MOF_CORE_SRC "${src_file}")
    endif()
endforeach()

# ------------------------------------------------------------
# Core library (preprocessing / features / modeling / evaluation / utils)
# If no non-main sources exist yet, use an INTERFACE target so config still works.
# ------------------------------------------------------------
if (MOF_CORE_SRC)
    add_library(mof_core STATIC ${MOF_CORE_SRC})
    target_include_directories(mof_core PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
    target_compile_features(mof_core PUBLIC cxx_std_${CMAKE_CXX_STANDARD})
    mof_enable_warnings(mof_core)
else()
    add_library(mof_core INTERFACE)
    target_include_directories(mof_core INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )
endif()

# ------------------------------------------------------------
# Main executable
# Fallback stub is generated only if src/main.cpp is not present yet.
# This keeps the project buildable from day one.
# ------------------------------------------------------------
if (MOF_MAIN_SRC)
    add_executable(mof_prediction_tool "${MOF_MAIN_SRC}")
else()
    set(MOF_STUB_MAIN "${CMAKE_CURRENT_BINARY_DIR}/generated_stub_main.cpp")
    file(WRITE "${MOF_STUB_MAIN}" [=[
#include <iostream>

int main() {
    std::cout << "MOF Property Prediction Tool build is set up.\n";
    std::cout << "Add src/main.cpp to run the full pipeline "
                 "(preprocessing -> features -> modeling -> evaluation).\n";
    return 0;
}
]=])
    add_executable(mof_prediction_tool "${MOF_STUB_MAIN}")
endif()

target_link_libraries(mof_prediction_tool PRIVATE mof_core)
target_include_directories(mof_prediction_tool PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_compile_features(mof_prediction_tool PRIVATE cxx_std_${CMAKE_CXX_STANDARD})
mof_enable_warnings(mof_prediction_tool)

# ------------------------------------------------------------
# Tests (future-ready, no external framework assumed)
# If your test files use a framework later (e.g., Catch2/GTest),
# add find_package(...) and link it here.
# ------------------------------------------------------------
include(CTest) # defines BUILD_TESTING option

if (BUILD_TESTING AND MOF_BUILD_TESTS)
    file(GLOB MOF_TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    )

    if (MOF_TEST_SOURCES)
        foreach(test_src IN LISTS MOF_TEST_SOURCES)
            get_filename_component(test_name "${test_src}" NAME_WE)

            add_executable("${test_name}" "${test_src}")
            target_link_libraries("${test_name}" PRIVATE mof_core)
            target_include_directories("${test_name}" PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/include"
            )
            target_compile_features("${test_name}" PRIVATE cxx_std_${CMAKE_CXX_STANDARD})
            mof_enable_warnings("${test_name}")

            # Assumes each test source builds to a runnable test executable.
            add_test(NAME "${test_name}" COMMAND "${test_name}")
        endforeach()
    else()
        message(STATUS "MOF_BUILD_TESTS=ON, but no tests/*.cpp files found yet.")
    endif()

    # --------------------------------------------------------
    # Optional explicit manual pipeline test registration
    # (matches requested add_executable/target_link_libraries lines)
    # Safe because tests/*.cpp may already auto-create this target.
    # --------------------------------------------------------
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/manual_pipeline_test.cpp" AND NOT TARGET manual_pipeline_test)
        add_executable(manual_pipeline_test tests/manual_pipeline_test.cpp)
        target_link_libraries(manual_pipeline_test PRIVATE mof_core)
        target_include_directories(manual_pipeline_test PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
        )
        target_compile_features(manual_pipeline_test PRIVATE cxx_std_${CMAKE_CXX_STANDARD})
        mof_enable_warnings(manual_pipeline_test)

        add_test(NAME manual_pipeline_test COMMAND manual_pipeline_test)
    endif()
endif()

# ------------------------------------------------------------
# Helpful status output
# ------------------------------------------------------------
message(STATUS "Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Main source: ${MOF_MAIN_SRC}")
message(STATUS "Core sources count: ${MOF_CORE_SRC}")
message(STATUS "Build tests: ${MOF_BUILD_TESTS} (BUILD_TESTING=${BUILD_TESTING})")