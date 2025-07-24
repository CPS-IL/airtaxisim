# Add libraries
add_library(
	dc_utils SHARED
	bbox.cpp
	cloud.cpp
	folder_reader.cpp
	rich_point.cpp
	velodyne_utils.cpp
)

# Set header targets
file(GLOB DC_UTILS_HEADERS "*.h" "*.hpp")

# Set library target properties
set_target_properties(
	dc_utils PROPERTIES
	PUBLIC_HEADER "${DC_UTILS_HEADERS}"
)

# Set library link targets
target_link_libraries(
	dc_utils
	${Boost_LIBRARIES}
	${OpenCV_LIBS}
)

# Set installation targets
install(
	TARGETS dc_utils
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include/depth_clustering/utils
)