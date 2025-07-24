# Add libraries
add_library(
	dc_post_processing SHARED
	bounding_box.cpp
	camera_projection.cpp
	logger.cpp
	score.cpp
)

# Set header targets
file(GLOB DC_POST_PROCESSING_HEADERS "*.h" "*.hpp")

# Set library target properties
set_target_properties(
	dc_post_processing PROPERTIES
	PUBLIC_HEADER "${DC_POST_PROCESSING_HEADERS}"
)

# Set library link targets
target_link_libraries(
	dc_post_processing
	${Boost_LIBRARIES}
)

# Set installation targets
install(
	TARGETS dc_post_processing
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include/depth_clustering/post_processing
)