/*
 * parameter_factory.cpp
 *
 *  Created on: Sep 10, 2020
 *      Author: simonyu
 */

#include <boost/property_tree/json_parser.hpp>

#include "api/parameter_factory.h"
#include "image_labelers/diff_helpers/diff_factory.h"
#include "projections/projection_params.h"

namespace depth_clustering
{
using boost::property_tree::json_parser::read_json;

ParameterFactory::ParameterFactory(const std::string& file_path_name_config)
{
	try
	{
		boost::property_tree::read_json(file_path_name_config, top_tree_);
	} catch (const std::exception &e)
	{
		std::cout << "[ERROR]: \"" << e.what() << "\"." << std::endl;
		std::cout << "[ERROR]: Failed to load configuration file: \"" << file_path_name_config
				<< "\"." << std::endl;
		return;
	}

	depth_clustering_tree_ = top_tree_.get_child_optional("depth_clustering");
	lidar_projection_tree_ = top_tree_.get_child_optional("lidar_projection");
	camera_projection_tree_ = top_tree_.get_child_optional("camera_projection");
	logger_tree_ = top_tree_.get_child_optional("logger");
}

DepthClusteringParameter
ParameterFactory::getDepthClusteringParameter()
{
	if (!depth_clustering_tree_)
	{
		std::cout << "[WARN]: Depth clustering configuration missing." << std::endl;
		return DepthClusteringParameter();
	}

	DepthClusteringParameter parameter;

	auto tree = *depth_clustering_tree_;

	auto distance_clustering_optional = tree.get_optional<float>("distance_clustering");
	auto score_clustering_optional = tree.get_optional<float>("score_clustering");
	auto angle_clustering_optional = tree.get_optional<float>("angle_clustering");
	auto angle_ground_removal_optional = tree.get_optional<float>("angle_ground_removal");
	auto size_cluster_min_optional = tree.get_optional<int>("size_cluster_min");
	auto size_cluster_max_optional = tree.get_optional<int>("size_cluster_max");
	auto size_smooth_window_optional = tree.get_optional<int>("size_smooth_window");
	auto use_camera_fov_optional = tree.get_optional<bool>("use_camera_fov");
	auto use_score_filter_optional = tree.get_optional<bool>("use_score_filter");
	auto score_filter_threshold_optional = tree.get_optional<float>("score_filter_threshold");
	auto score_type_point_optional = tree.get_optional<std::string>("score_type_point");
	auto score_type_cluster_optional = tree.get_optional<std::string>("score_type_cluster");
	auto score_type_frame_optional = tree.get_optional<std::string>("score_type_frame");
	auto bounding_box_type_optional = tree.get_optional<std::string>("bounding_box_type");
	auto difference_type_optional = tree.get_optional<std::string>("difference_type");
	auto dataset_file_type_optional = tree.get_optional<std::string>("dataset_file_type");
	auto dataset_name_optional = tree.get_optional<std::string>("dataset_name");
	auto ground_truth_cube_file_name_optional = tree.get_optional<std::string>(
			"ground_truth_cube_file_name");
	auto ground_truth_flat_file_name_optional = tree.get_optional<std::string>(
			"ground_truth_flat_file_name");

	if (distance_clustering_optional)
	{
		parameter.distance_clustering = *distance_clustering_optional;
	}

	if (score_clustering_optional)
	{
		parameter.score_clustering = *score_clustering_optional;
	}

	if (angle_clustering_optional)
	{
		parameter.angle_clustering = Radians
		{ Radians::IsRadians
		{ }, static_cast<float>(*angle_clustering_optional * M_PI / 180.0) };
	}

	if (angle_ground_removal_optional)
	{
		parameter.angle_ground_removal = Radians
		{ Radians::IsRadians
		{ }, static_cast<float>(*angle_ground_removal_optional * M_PI / 180.0) };
	}

	if (size_cluster_min_optional)
	{
		parameter.size_cluster_min = *size_cluster_min_optional;
	}

	if (size_cluster_max_optional)
	{
		parameter.size_cluster_max = *size_cluster_max_optional;
	}

	if (size_smooth_window_optional)
	{
		parameter.size_smooth_window = *size_smooth_window_optional;
	}

	if (use_camera_fov_optional)
	{
		parameter.use_camera_fov = *use_camera_fov_optional;
	}

	if (use_score_filter_optional)
	{
		parameter.use_score_filter = *use_score_filter_optional;
	}

	if (score_filter_threshold_optional)
	{
		parameter.score_filter_threshold = *score_filter_threshold_optional;
	}

	if (score_type_point_optional)
	{
		std::string score_type_point_string = *score_type_point_optional;

		if (score_type_point_string == "type_1")
		{
			parameter.score_type_point = Score::TypePoint::Type_1;
		}
		else if (score_type_point_string == "type_2")
		{
			parameter.score_type_point = Score::TypePoint::Type_2;
		}
		else
		{
			std::cout << "[WARN]: Unknown point score type." << std::endl;
			parameter.score_type_point = Score::TypePoint::Type_1;
		}
	}

	if (score_type_cluster_optional)
	{
		std::string score_type_cluster_string = *score_type_cluster_optional;

		if (score_type_cluster_string == "type_1")
		{
			parameter.score_type_cluster = Score::TypeCluster::Type_1;
		}
		else if (score_type_cluster_string == "type_2")
		{
			parameter.score_type_cluster = Score::TypeCluster::Type_2;
		}
		else
		{
			std::cout << "[WARN]: Unknown cluster score type." << std::endl;
			parameter.score_type_cluster = Score::TypeCluster::Type_1;
		}
	}

	if (score_type_frame_optional)
	{
		std::string score_type_frame_string = *score_type_frame_optional;

		if (score_type_frame_string == "type_1")
		{
			parameter.score_type_frame = Score::TypeFrame::Type_1;
		}
		else
		{
			std::cout << "[WARN]: Unknown frame score type." << std::endl;
			parameter.score_type_frame = Score::TypeFrame::Type_1;
		}
	}

	if (bounding_box_type_optional)
	{
		std::string bounding_box_type_string = *bounding_box_type_optional;

		if (bounding_box_type_string == "cube")
		{
			parameter.bounding_box_type = BoundingBox::Type::Cube;
		}
		else if (bounding_box_type_string == "polygon")
		{
			parameter.bounding_box_type = BoundingBox::Type::Polygon;
		}
		else
		{
			std::cout << "[WARN]: Unknown bounding box type." << std::endl;
			parameter.bounding_box_type = BoundingBox::Type::Polygon;
		}
	}

	if (difference_type_optional)
	{
		std::string difference_type_string = *difference_type_optional;

		if (difference_type_string == "angles")
		{
			parameter.difference_type = DiffFactory::DiffType::ANGLES;
		}
		else if (difference_type_string == "angles_precomputed")
		{
			parameter.difference_type = DiffFactory::DiffType::ANGLES_PRECOMPUTED;
		}
		else if (difference_type_string == "line_dist")
		{
			parameter.difference_type = DiffFactory::DiffType::LINE_DIST;
		}
		else if (difference_type_string == "line_dist_precomputed")
		{
			parameter.difference_type = DiffFactory::DiffType::LINE_DIST_PRECOMPUTED;
		}
		else if (difference_type_string == "simple")
		{
			parameter.difference_type = DiffFactory::DiffType::SIMPLE;
		}
		else
		{
			std::cout << "[WARN]: Unknown clustering type." << std::endl;
			parameter.difference_type = DiffFactory::DiffType::ANGLES_PRECOMPUTED;
		}
	}

	if (dataset_file_type_optional)
	{
		parameter.dataset_file_type = *dataset_file_type_optional;
	}

	if (dataset_name_optional)
	{
		parameter.dataset_name = *dataset_name_optional;
	}

	if (ground_truth_cube_file_name_optional)
	{
		parameter.ground_truth_cube_file_name = *ground_truth_cube_file_name_optional;
	}

	if (ground_truth_flat_file_name_optional)
	{
		parameter.ground_truth_flat_file_name = *ground_truth_flat_file_name_optional;
	}

	return parameter;
}

std::shared_ptr<ProjectionParams>
ParameterFactory::getLidarProjectionParameter()
{
	if (!lidar_projection_tree_)
	{
		std::cout << "[WARN]: Lidar projection configuration missing." << std::endl;
		return std::unique_ptr<ProjectionParams>(new ProjectionParams());
	}

	auto projection_parameter_raw = std::make_shared<ProjectionParamsRaw>();

	auto tree = *lidar_projection_tree_;

	auto horizontal_steps_optional = tree.get_optional<int>("horizontal_steps");
	auto beams_optional = tree.get_optional<int>("beams");
	auto horizontal_angle_start_optional = tree.get_optional<int>("horizontal_angle_start");
	auto horizontal_angle_end_optional = tree.get_optional<int>("horizontal_angle_end");
	auto intensity_norm_factor_optional = tree.get_optional<double>("intensity_norm_factor");
	auto elongation_norm_factor_optional = tree.get_optional<double>("elongation_norm_factor");
	auto beam_inclinations_optional = tree.get_child_optional("beam_inclinations");
	auto extrinsic_optional = tree.get_child_optional("extrinsic");

	if (horizontal_steps_optional)
	{
		projection_parameter_raw->horizontal_steps = *horizontal_steps_optional;
	}

	if (beams_optional)
	{
		projection_parameter_raw->beams = *beams_optional;
	}

	if (horizontal_angle_start_optional)
	{
		projection_parameter_raw->horizontal_angle_start = *horizontal_angle_start_optional;
	}

	if (horizontal_angle_end_optional)
	{
		projection_parameter_raw->horizontal_angle_end = *horizontal_angle_end_optional;
	}

	if (intensity_norm_factor_optional)
	{
		projection_parameter_raw->intensity_norm_factor = *intensity_norm_factor_optional;
	}

	if (elongation_norm_factor_optional)
	{
		projection_parameter_raw->elongation_norm_factor = *elongation_norm_factor_optional;
	}

	if (beam_inclinations_optional)
	{
		projection_parameter_raw->beam_inclinations.clear();

		for (const auto &beam_inclinations_pair : *beam_inclinations_optional)
		{
			projection_parameter_raw->beam_inclinations.push_back(
					beam_inclinations_pair.second.get_value<double>());
		}
	}

	if (extrinsic_optional)
	{
		projection_parameter_raw->extrinsic.clear();

		for (const auto &extrinsic_pair : *extrinsic_optional)
		{
			projection_parameter_raw->extrinsic.push_back(
					extrinsic_pair.second.get_value<double>());
		}
	}

	projection_parameter_raw->updateHorizontalAngles(
			projection_parameter_raw->horizontal_angle_start,
			projection_parameter_raw->horizontal_angle_end);

	auto projection_parameter = ProjectionParams::FromBeamInclinations(
			projection_parameter_raw->horizontal_steps, projection_parameter_raw->beams,
			projection_parameter_raw->horizontal_angle_start,
			projection_parameter_raw->horizontal_angle_end,
			projection_parameter_raw->beam_inclinations);

	projection_parameter->setProjectionParamsRaw(projection_parameter_raw);

	return std::move(projection_parameter);
}

CameraProjectionParameter
ParameterFactory::getCameraProjectionParameter()
{
	if (!camera_projection_tree_)
	{
		std::cout << "[WARN]: Camera projection configuration missing." << std::endl;
		return CameraProjectionParameter();
	}

	CameraProjectionParameter parameter;

	auto tree = *camera_projection_tree_;

	auto intrinsic_optional = tree.get_child_optional("intrinsic");
	auto extrinsic_optional = tree.get_child_optional("extrinsic");
	auto width_optional = tree.get_optional<int>("width");
	auto height_optional = tree.get_optional<int>("height");
	auto field_of_view_angle_start_optional = tree.get_optional<int>("field_of_view_angle_start");
	auto field_of_view_angle_end_optional = tree.get_optional<int>("field_of_view_angle_end");
	auto threshold_truncation_optional = tree.get_optional<double>("threshold_truncation");
	auto threshold_filter_height_optional = tree.get_optional<double>("threshold_filter_height");
	auto threshold_filter_tunnel_left_optional = tree.get_optional<double>(
			"threshold_filter_tunnel_left");
	auto threshold_filter_tunnel_right_optional = tree.get_optional<double>(
			"threshold_filter_tunnel_right");
	auto threshold_filter_tunnel_front_optional = tree.get_optional<double>(
			"threshold_filter_tunnel_front");
	auto correct_distortions_optional = tree.get_optional<bool>("correct_distortions");
	auto use_filter_height_optional = tree.get_optional<bool>("use_filter_height");
	auto use_filter_tunnel_optional = tree.get_optional<bool>("use_filter_tunnel");

	if (intrinsic_optional)
	{
		for (const auto &intrinsic_pair : *intrinsic_optional)
		{
			parameter.intrinsic.push_back(intrinsic_pair.second.get_value<double>());
		}
	}

	if (extrinsic_optional)
	{
		for (const auto &extrinsic_pair : *extrinsic_optional)
		{
			parameter.extrinsic.push_back(extrinsic_pair.second.get_value<double>());
		}
	}

	if (width_optional)
	{
		parameter.width = *width_optional;
	}

	if (height_optional)
	{
		parameter.height = *height_optional;
	}

	if (field_of_view_angle_start_optional)
	{
		parameter.field_of_view_angle_start = *field_of_view_angle_start_optional;
	}

	if (field_of_view_angle_end_optional)
	{
		parameter.field_of_view_angle_end = *field_of_view_angle_end_optional;
	}

	if (threshold_truncation_optional)
	{
		parameter.threshold_truncation = *threshold_truncation_optional;
	}

	if (threshold_filter_height_optional)
	{
		parameter.threshold_filter_height = *threshold_filter_height_optional;
	}

	if (threshold_filter_tunnel_left_optional)
	{
		parameter.threshold_filter_tunnel_left = *threshold_filter_tunnel_left_optional;
	}

	if (threshold_filter_tunnel_right_optional)
	{
		parameter.threshold_filter_tunnel_right = *threshold_filter_tunnel_right_optional;
	}

	if (threshold_filter_tunnel_front_optional)
	{
		parameter.threshold_filter_tunnel_front = *threshold_filter_tunnel_front_optional;
	}

	if (correct_distortions_optional)
	{
		parameter.correct_distortions = *correct_distortions_optional;
	}

	if (use_filter_height_optional)
	{
		parameter.use_filter_height = *use_filter_height_optional;
	}

	if (use_filter_tunnel_optional)
	{
		parameter.use_filter_tunnel = *use_filter_tunnel_optional;
	}

	return parameter;
}

LoggerParameter
ParameterFactory::getLoggerParameter()
{
	if (!logger_tree_)
	{
		std::cout << "[WARN]: Logger configuration missing." << std::endl;
		return LoggerParameter();
	}

	LoggerParameter parameter;

	auto tree = *logger_tree_;

	auto log_path_optional = tree.get_optional<std::string>("log_path");
	auto log_file_name_cube_optional = tree.get_optional<std::string>("log_file_name_cube");
	auto log_file_name_polygon_optional = tree.get_optional<std::string>("log_file_name_polygon");
	auto log_file_name_flat_optional = tree.get_optional<std::string>("log_file_name_flat");
	auto log_optional = tree.get_optional<bool>("log");

	if (log_path_optional)
	{
		parameter.log_path = *log_path_optional;
	}

	if (log_file_name_cube_optional)
	{
		parameter.log_file_name_cube = *log_file_name_cube_optional;
	}

	if (log_file_name_polygon_optional)
	{
		parameter.log_file_name_polygon = *log_file_name_polygon_optional;
	}

	if (log_file_name_flat_optional)
	{
		parameter.log_file_name_flat = *log_file_name_flat_optional;
	}

	if (log_optional)
	{
		parameter.log = *log_optional;
	}

	return parameter;
}

void
ParameterFactory::setGlobalDepthClusteringParameter(DepthClusteringParameter& parameter)
{
	if (!depth_clustering_tree_)
	{
		std::cout << "[WARN]: Depth clustering configuration missing." << std::endl;
		return;
	}

	auto tree = *depth_clustering_tree_;

	auto distance_clustering_optional = tree.get_optional<float>("distance_clustering");
	auto score_clustering_optional = tree.get_optional<float>("score_clustering");
	auto angle_clustering_optional = tree.get_optional<float>("angle_clustering");
	auto angle_ground_removal_optional = tree.get_optional<float>("angle_ground_removal");
	auto size_cluster_min_optional = tree.get_optional<int>("size_cluster_min");
	auto size_cluster_max_optional = tree.get_optional<int>("size_cluster_max");
	auto size_smooth_window_optional = tree.get_optional<int>("size_smooth_window");
	auto use_camera_fov_optional = tree.get_optional<bool>("use_camera_fov");
	auto use_score_filter_optional = tree.get_optional<bool>("use_score_filter");
	auto score_filter_threshold_optional = tree.get_optional<float>("score_filter_threshold");
	auto score_type_point_optional = tree.get_optional<std::string>("score_type_point");
	auto score_type_cluster_optional = tree.get_optional<std::string>("score_type_cluster");
	auto score_type_frame_optional = tree.get_optional<std::string>("score_type_frame");
	auto bounding_box_type_optional = tree.get_optional<std::string>("bounding_box_type");
	auto difference_type_optional = tree.get_optional<std::string>("difference_type");
	auto ground_truth_cube_file_name_optional = tree.get_optional<std::string>(
			"ground_truth_cube_file_name");
	auto ground_truth_flat_file_name_optional = tree.get_optional<std::string>(
			"ground_truth_flat_file_name");

	if (distance_clustering_optional)
	{
		parameter.distance_clustering = *distance_clustering_optional;
	}

	if (score_clustering_optional)
	{
		parameter.score_clustering = *score_clustering_optional;
	}

	if (angle_clustering_optional)
	{
		parameter.angle_clustering = Radians
		{ Radians::IsRadians
		{ }, static_cast<float>(*angle_clustering_optional * M_PI / 180.0) };
	}

	if (angle_ground_removal_optional)
	{
		parameter.angle_ground_removal = Radians
		{ Radians::IsRadians
		{ }, static_cast<float>(*angle_ground_removal_optional * M_PI / 180.0) };
	}

	if (size_cluster_min_optional)
	{
		parameter.size_cluster_min = *size_cluster_min_optional;
	}

	if (size_cluster_max_optional)
	{
		parameter.size_cluster_max = *size_cluster_max_optional;
	}

	if (size_smooth_window_optional)
	{
		parameter.size_smooth_window = *size_smooth_window_optional;
	}

	if (use_camera_fov_optional)
	{
		parameter.use_camera_fov = *use_camera_fov_optional;
	}

	if (use_score_filter_optional)
	{
		parameter.use_score_filter = *use_score_filter_optional;
	}

	if (score_filter_threshold_optional)
	{
		parameter.score_filter_threshold = *score_filter_threshold_optional;
	}

	if (score_type_point_optional)
	{
		std::string score_type_point_string = *score_type_point_optional;

		if (score_type_point_string == "type_1")
		{
			parameter.score_type_point = Score::TypePoint::Type_1;
		}
		else if (score_type_point_string == "type_2")
		{
			parameter.score_type_point = Score::TypePoint::Type_2;
		}
		else
		{
			std::cout << "[WARN]: Unknown point score type." << std::endl;
			parameter.score_type_point = Score::TypePoint::Type_1;
		}
	}

	if (score_type_cluster_optional)
	{
		std::string score_type_cluster_string = *score_type_cluster_optional;

		if (score_type_cluster_string == "type_1")
		{
			parameter.score_type_cluster = Score::TypeCluster::Type_1;
		}
		else if (score_type_cluster_string == "type_2")
		{
			parameter.score_type_cluster = Score::TypeCluster::Type_2;
		}
		else
		{
			std::cout << "[WARN]: Unknown cluster score type." << std::endl;
			parameter.score_type_cluster = Score::TypeCluster::Type_1;
		}
	}

	if (score_type_frame_optional)
	{
		std::string score_type_frame_string = *score_type_frame_optional;

		if (score_type_frame_string == "type_1")
		{
			parameter.score_type_frame = Score::TypeFrame::Type_1;
		}
		else
		{
			std::cout << "[WARN]: Unknown frame score type." << std::endl;
			parameter.score_type_frame = Score::TypeFrame::Type_1;
		}
	}

	if (bounding_box_type_optional)
	{
		std::string bounding_box_type_string = *bounding_box_type_optional;

		if (bounding_box_type_string == "cube")
		{
			parameter.bounding_box_type = BoundingBox::Type::Cube;
		}
		else if (bounding_box_type_string == "polygon")
		{
			parameter.bounding_box_type = BoundingBox::Type::Polygon;
		}
		else
		{
			std::cout << "[WARN]: Unknown bounding box type." << std::endl;
			parameter.bounding_box_type = BoundingBox::Type::Polygon;
		}
	}

	if (difference_type_optional)
	{
		std::string difference_type_string = *difference_type_optional;

		if (difference_type_string == "angles")
		{
			parameter.difference_type = DiffFactory::DiffType::ANGLES;
		}
		else if (difference_type_string == "angles_precomputed")
		{
			parameter.difference_type = DiffFactory::DiffType::ANGLES_PRECOMPUTED;
		}
		else if (difference_type_string == "line_dist")
		{
			parameter.difference_type = DiffFactory::DiffType::LINE_DIST;
		}
		else if (difference_type_string == "line_dist_precomputed")
		{
			parameter.difference_type = DiffFactory::DiffType::LINE_DIST_PRECOMPUTED;
		}
		else if (difference_type_string == "simple")
		{
			parameter.difference_type = DiffFactory::DiffType::SIMPLE;
		}
		else
		{
			std::cout << "[WARN]: Unknown clustering type." << std::endl;
			parameter.difference_type = DiffFactory::DiffType::ANGLES_PRECOMPUTED;
		}
	}

	if (ground_truth_cube_file_name_optional)
	{
		parameter.ground_truth_cube_file_name = *ground_truth_cube_file_name_optional;
	}

	if (ground_truth_flat_file_name_optional)
	{
		parameter.ground_truth_flat_file_name = *ground_truth_flat_file_name_optional;
	}
}

void
ParameterFactory::setGlobalLidarProjectionParameter(
		std::shared_ptr<ProjectionParams> parameter_projection_lidar)
{
	// No global lidar projection parameters for now
}

void
ParameterFactory::setGlobalCameraProjectionParameter(
		CameraProjectionParameter& parameter_projection_camera)
{
	// No global camera projection parameters for now
}

void
ParameterFactory::setGlobalLoggerParameter(LoggerParameter& parameter_logger)
{
	if (!logger_tree_)
	{
		std::cout << "[WARN]: Logger configuration missing." << std::endl;
		return;
	}

	auto tree = *logger_tree_;

	auto log_path_optional = tree.get_optional<std::string>("log_path");
	auto log_file_name_cube_optional = tree.get_optional<std::string>("log_file_name_cube");
	auto log_file_name_polygon_optional = tree.get_optional<std::string>("log_file_name_polygon");
	auto log_file_name_flat_optional = tree.get_optional<std::string>("log_file_name_flat");
	auto log_optional = tree.get_optional<bool>("log");

	if (log_path_optional)
	{
		parameter_logger.log_path = *log_path_optional;
	}

	if (log_file_name_cube_optional)
	{
		parameter_logger.log_file_name_cube = *log_file_name_cube_optional;
	}

	if (log_file_name_polygon_optional)
	{
		parameter_logger.log_file_name_polygon = *log_file_name_polygon_optional;
	}

	if (log_file_name_flat_optional)
	{
		parameter_logger.log_file_name_flat = *log_file_name_flat_optional;
	}

	if (log_optional)
	{
		parameter_logger.log = *log_optional;
	}
}
} // namespace depth_clustering
