/*
 * parameter_factory.h
 *
 *  Created on: Sep 10, 2020
 *      Author: simonyu
 */

#ifndef SRC_API_PARAMETER_FACTORY_H_
#define SRC_API_PARAMETER_FACTORY_H_

#include <boost/property_tree/ptree.hpp>

#include "api/parameter.h"
#include "post_processing/logger_parameter.h"
#include "projections/projection_params.h"

namespace depth_clustering
{
class ParameterFactory
{
public:

	ParameterFactory(const std::string& file_path_name_config);

	DepthClusteringParameter
	getDepthClusteringParameter();

	std::shared_ptr<ProjectionParams>
	getLidarProjectionParameter();

	CameraProjectionParameter
	getCameraProjectionParameter();

	LoggerParameter
	getLoggerParameter();

	void
	setGlobalDepthClusteringParameter(DepthClusteringParameter& parameter);

	void
	setGlobalLidarProjectionParameter(std::shared_ptr<ProjectionParams> parameter_projection_lidar);

	void
	setGlobalCameraProjectionParameter(CameraProjectionParameter& parameter_projection_camera);

	void
	setGlobalLoggerParameter(LoggerParameter& parameter_logger);

private:

	boost::property_tree::ptree top_tree_;
	boost::optional<boost::property_tree::ptree> depth_clustering_tree_;
	boost::optional<boost::property_tree::ptree> lidar_projection_tree_;
	boost::optional<boost::property_tree::ptree> camera_projection_tree_;
	boost::optional<boost::property_tree::ptree> logger_tree_;
};
} // namespace depth_clustering

#endif /* SRC_API_PARAMETER_FACTORY_H_ */
