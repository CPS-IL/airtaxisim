# Add libraries
add_library(
	dc_image_labelers SHARED
	diff_helpers/angle_diff.cpp
	diff_helpers/line_dist_diff.cpp
	abstract_image_labeler.cpp
)

# Set header targets
file(GLOB DC_IMAGE_LABELERS_HEADERS "*.h" "*.hpp")

# Set library target properties
set_target_properties(
	dc_image_labelers PROPERTIES
	PUBLIC_HEADER "${DC_IMAGE_LABELERS_HEADERS}"
)

# Set installation targets
install(
	TARGETS dc_image_labelers
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include/depth_clustering/image_labelers
)

# Add subdirectories
add_subdirectory(diff_helpers)