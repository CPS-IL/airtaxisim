# Add libraries
add_library(
	dc_communication SHARED
	identifiable.cpp
)

# Set header targets
file(GLOB DC_COMMUNICATION_HEADERS "*.h" "*.hpp")

# Set library target properties
set_target_properties(
	dc_communication PROPERTIES
	PUBLIC_HEADER "${DC_COMMUNICATION_HEADERS}"
)

# Set installation targets
install(
	TARGETS dc_communication
	LIBRARY DESTINATION lib
	PUBLIC_HEADER DESTINATION include/depth_clustering/communication
)