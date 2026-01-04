cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(AIImageClassifier)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find LibTorch
# You can specify the path to LibTorch by setting CMAKE_PREFIX_PATH
# Example: cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/ImageClassifier.cpp
)

# Create executable
add_executable(image_classifier ${SOURCES})

# Link libraries
target_link_libraries(image_classifier 
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBS}
)

# Set properties for proper linking
set_property(TARGET image_classifier PROPERTY CXX_STANDARD 17)

# On Windows, copy DLLs to output directory
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET image_classifier
                       POST_BUILD
                       COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${TORCH_DLLS}
                       $<TARGET_FILE_DIR:image_classifier>)
endif (MSVC)

# Print configuration information
message(STATUS "")
message(STATUS "=== Build Configuration ===")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Torch Version: ${TORCH_VERSION}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
message(STATUS "===========================")
message(STATUS "")
