/*
 * logger.h
 *
 *  Created on: Sep 9, 2020
 *      Author: simonyu
 */

#ifndef SRC_POST_PROCESSING_LOGGER_H_
#define SRC_POST_PROCESSING_LOGGER_H_

#include <boost/property_tree/ptree.hpp>

#include "post_processing/camera_projection.h"
#include "post_processing/bounding_box.h"
#include "post_processing/logger_parameter.h"

namespace depth_clustering
{

class Logger
{
public:

	Logger();

	Logger(const LoggerParameter& parameter);

	void
	setBoundingBox(std::shared_ptr<BoundingBox> bounding_box);

	void
	logBoundingBoxFrame(const std::string& frame_name, const BoundingBox::Type& bounding_box_type);

	void
	logBoundingBoxFrameCube(const std::string& frame_name);

	void
	logBoundingBoxFramePolygon(const std::string& frame_name);

	void
	logBoundingBoxFrameFlat(const std::string& frame_name);

	void
	writeBoundingBoxLog(const BoundingBox::Type& bounding_box_type);

private:

	LoggerParameter parameter_;
	boost::property_tree::ptree bounding_box_log_tree_cube_;
	boost::property_tree::ptree bounding_box_log_tree_polygon_;
	boost::property_tree::ptree bounding_box_log_tree_flat_;

	std::shared_ptr<BoundingBox> bounding_box_;
};

} // namespace depth_clustering

#endif /* SRC_POST_PROCESSING_LOGGER_H_ */
