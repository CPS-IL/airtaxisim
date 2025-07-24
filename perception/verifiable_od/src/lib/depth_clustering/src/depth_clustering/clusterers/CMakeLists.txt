# Set installation targets
install(
	DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include/depth_clustering FILES_MATCHING
	PATTERN "*.h"
	PATTERN "*.hpp"
)