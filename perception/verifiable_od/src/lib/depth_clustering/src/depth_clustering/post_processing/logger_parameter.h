/*
 * logger_parameter.h
 *
 *  Created on: Sep 11, 2020
 *      Author: simonyu
 */

#ifndef SRC_POST_PROCESSING_LOGGER_PARAMETER_H_
#define SRC_POST_PROCESSING_LOGGER_PARAMETER_H_

#include <string>

struct LoggerParameter
{
	std::string log_path;
	std::string log_file_name_cube;
	std::string log_file_name_polygon;
	std::string log_file_name_flat;
	bool log;

	LoggerParameter() :
			log_path("./"), log_file_name_cube("depth_clustering_detection_cube.json"), log_file_name_polygon(
					"depth_clustering_detection_polygon.json"), log_file_name_flat(
					"depth_clustering_detection_flat.json"), log(true)
	{
	}
};

#endif /* SRC_POST_PROCESSING_LOGGER_PARAMETER_H_ */
