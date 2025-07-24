# Add libraries
add_library(
	dc_ground_removal SHARED
	depth_ground_remover.cpp
)

# Set header targets
file(GLOB DC_GROUND_REMOVAL_HEADERS "*.h" "*.hpp")

# Set library target properties
set_target_properties(
	dc_ground_removal PROPERTIES
	PUBLIC_HEADER "${DC_GROUND_REMOVAL_HEADERS}"
)

# Set library link targets
target_link_libraries(
	dc_ground_removal
	dc_communication
	dc_image_labelers
	dc_utils
	${OpenCV_LIBS}
)

# Set installation targets
install(
	TARGETS dc_ground_removal
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include/depth_clustering/ground_removal
)