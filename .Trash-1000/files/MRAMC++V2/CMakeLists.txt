cmake_minimum_required(VERSION 3.10)
project(EnhancedMarketingAgent VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler flags
if(MSVC)
    # MSVC compiler flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
else()
    # GCC/Clang compiler flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Source files for the library
set(LIBRARY_SOURCES
    enhanced_marketing_agent.cpp
)

set(LIBRARY_HEADERS
    enhanced_marketing_agent.hpp
)

# Create static library
add_library(marketing_agent STATIC
    ${LIBRARY_SOURCES}
    ${LIBRARY_HEADERS}
)

target_include_directories(marketing_agent PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Create demo executable
add_executable(marketing_demo
    demo.cpp
)

target_link_libraries(marketing_demo PRIVATE
    marketing_agent
)

# Optional: Enable testing
option(BUILD_TESTS "Build test suite" OFF)
if(BUILD_TESTS)
    enable_testing()
    # Add test subdirectory here if you create tests
endif()

# Installation rules
install(TARGETS marketing_agent marketing_demo
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${LIBRARY_HEADERS}
    DESTINATION include
)

# Print configuration info
message(STATUS "")
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "  Enhanced Marketing Agent - Build Configuration")
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "  Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler:      ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Standard:      C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Install prefix:    ${CMAKE_INSTALL_PREFIX}")
message(STATUS "═══════════════════════════════════════════════════════")
message(STATUS "")
