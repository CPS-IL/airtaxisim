# Add libraries
add_library(
	dc_api SHARED
	api.cpp
	parameter_factory.cpp
)

# Set header targets
file(GLOB DC_API_HEADERS "*.h" "*.hpp")

# Set library target properties
set_target_properties(
	dc_api PROPERTIES
	PUBLIC_HEADER "${DC_API_HEADERS}"
)

# Set library link targets
target_link_libraries(
	dc_api
	dc_communication
	dc_ground_removal
	dc_image_labelers
	dc_post_processing
	dc_projections
	dc_utils
	${Boost_LIBRARIES}
	${OpenCV_LIBS}
)

# Set installation targets
install(
	TARGETS dc_api
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include/depth_clustering/api
)