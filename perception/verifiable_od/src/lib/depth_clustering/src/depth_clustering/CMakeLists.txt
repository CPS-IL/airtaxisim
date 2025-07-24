# Set global parameters
cmake_minimum_required(VERSION 2.8)
project(depth_clustering)

# Set CMake parameters
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../../cmake/depth_clustering")

# Set compiler parameters
set(CMAKE_C_FLAGS "-Wall -Wextra -static-libc -fPIC")
set(CMAKE_C_FLAGS_DEBUG "-g -O0")
set(CMAKE_C_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -Wextra -Wno-unused-parameter -Wno-int-in-bool-context -Wno-alloc-size-larger-than=9223372036854775807 -Wno-deprecated-copy -Wno-class-memaccess -Wno-parentheses -Wno-redundant-move -static-libstdc++ -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# arm64-only configurations
if(ARCH STREQUAL "arm64")
	set(CMAKE_EXE_LINKER_FLAGS "-Wl,--whole-archive")
endif()

# Set installation paths
string(TOLOWER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_PATH)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../../install/${ARCH}/depth_clustering/${CMAKE_BUILD_TYPE_PATH}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../../install/${ARCH}/depth_clustering/${CMAKE_BUILD_TYPE_PATH}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../../install/${ARCH}/depth_clustering/${CMAKE_BUILD_TYPE_PATH}/lib)

# Set package parameters
if(${LOCAL})
	set(Boost_NO_BOOST_CMAKE TRUE)
	set(Boost_NO_SYSTEM_PATHS TRUE)
	set(BOOST_ROOT ${PROJECT_SOURCE_DIR}/../../install/${ARCH}/boost)
	set(Boost_USE_STATIC_LIBS ON)
	set(EIGEN3_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../../install/${ARCH}/eigen/include/eigen3)
	set(OpenCV_STATIC ON)
endif()

# Find packages
if(${LOCAL})
	find_package(Boost REQUIRED COMPONENTS system filesystem regex program_options)
	find_package(Eigen3 REQUIRED)
	find_package(OpenCV REQUIRED PATHS ${PROJECT_SOURCE_DIR}/../../install/${ARCH}/opencv NO_DEFAULT_PATH)
else()
	find_package(Boost REQUIRED COMPONENTS system filesystem regex program_options)
	find_package(Eigen3 REQUIRED)
	find_package(OpenCV REQUIRED)
endif()

# Set include directories
if(${LOCAL})
	include_directories(
		${PROJECT_SOURCE_DIR}
		${PROJECT_BINARY_DIR}
		${Boost_INCLUDE_DIRS}
		${EIGEN3_INCLUDE_DIR}
		${OpenCV_INCLUDE_DIRS}
	)
else()
	include_directories(
		${PROJECT_SOURCE_DIR}
		${PROJECT_BINARY_DIR}
		SYSTEM ${Boost_INCLUDE_DIRS}
		SYSTEM ${EIGEN3_INCLUDE_DIR}
		SYSTEM ${OpenCV_INCLUDE_DIRS}
	)
endif()

# Set installation targets
install(
	DIRECTORY ${PROJECT_SOURCE_DIR}/../../config/
	DESTINATION config/depth_clustering
)
install(
	FILES ${PROJECT_SOURCE_DIR}/../../scripts/depth_clustering/depth_clustering.bash
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
	DESTINATION bin
)

# Add subdirectories
add_subdirectory(api)
add_subdirectory(applications)
add_subdirectory(clusterers)
add_subdirectory(communication)
add_subdirectory(ground_removal)
add_subdirectory(image_labelers)
add_subdirectory(post_processing)
add_subdirectory(projections)
add_subdirectory(utils)
