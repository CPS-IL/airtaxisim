# Add libraries
add_library(
	dc_projections SHARED
	cloud_projection.cpp
	projection_params.cpp
	ring_projection.cpp
	spherical_projection.cpp
)

# Set header targets
file(GLOB DC_PROJECTIONS_HEADERS "*.h" "*.hpp")

# Set library target properties
set_target_properties(
	dc_projections PROPERTIES
	PUBLIC_HEADER "${DC_PROJECTIONS_HEADERS}"
)

# Set library link targets
target_link_libraries(
	dc_projections
	${OpenCV_LIBS}
)

# Set installation targets
install(
	TARGETS dc_projections
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include/depth_clustering/projections
)