/*
 * api.cpp
 *
 *  Created on: Apr 27, 2020
 *      Author: simonyu
 */

#include <boost/property_tree/json_parser.hpp>

#include "api/api.h"
#include "api/parameter_factory.h"
#include "image_labelers/diff_helpers/diff_factory.h"
#include "utils/cloud.h"
#include "utils/radians.h"
#include "utils/rich_point.h"
#include "utils/timer.h"
#include "utils/velodyne_utils.h"

using boost::property_tree::json_parser::read_json;
using depth_clustering::time_utils::Timer;

namespace depth_clustering
{

DepthClustering::DepthClustering() :
		DepthClustering(DepthClusteringParameter())
{
}

DepthClustering::DepthClustering(const DepthClusteringParameter& parameter) :
		parameter_(parameter), dataset_path_("./"), frame_counter_(0)
{
}

const DepthClusteringParameter&
DepthClustering::getParameter() const
{
	return parameter_;
}

const CameraProjectionParameter&
DepthClustering::getCameraProjectionParameter() const
{
	return parameter_projection_camera_;
}

std::shared_ptr<ProjectionParams>
DepthClustering::getLidarProjectionParameter() const
{
	return parameter_projection_lidar_;
}

const std::string&
DepthClustering::getDatasetPath() const
{
	return dataset_path_;
}

Cloud::ConstPtr
DepthClustering::getCloud() const
{
	if (image_range_.rows == 0 || image_range_.cols == 0)
	{
		return nullptr;
	}

	return cloud_;
}

const cv::Mat
DepthClustering::getImageCamera(const std::string& frame_path_name_camera) const
{
	if (frame_path_name_camera == "")
	{
		std::cout << "[WARN]: Invalid camera frame path and name." << std::endl;
		return cv::Mat();
	}

	return MatFromPNGCamera(frame_path_name_camera);
}

const cv::Mat&
DepthClustering::getImageRange() const
{
	return image_range_;
}

const cv::Mat&
DepthClustering::getImageIntensity() const
{
	return image_intensity_;
}

const cv::Mat&
DepthClustering::getImageElongation() const
{
	return image_elongation_;
}

std::shared_ptr<Score>
DepthClustering::getScore() const
{
	return score_;
}

std::shared_ptr<BoundingBox>
DepthClustering::getBoundingBox() const
{
	return bounding_box_;
}

std::shared_ptr<BoundingBox::Frame<BoundingBox::Flat>>
DepthClustering::getBoundingBoxFrameFlat() const
{
	if (!bounding_box_)
	{
		return nullptr;
	}

	return bounding_box_->getFrameFlat();
}

std::shared_ptr<BoundingBox::Frame<BoundingBox::Cube>>
DepthClustering::getGroundTruthFrameCube(const std::string& frame_path_name_range)
{
	if (frame_path_name_range == "")
	{
		std::cout << "[WARN]: Invalid range frame path and name." << std::endl;
		return nullptr;
	}

	std::string frame_name_range = "";
	std::stringstream ss(frame_path_name_range);

	while (std::getline(ss, frame_name_range, '/'))
	{
	}

	if (frame_name_range == "")
	{
		std::cout << "[WARN]: Invalid range frame name." << std::endl;
		return nullptr;
	}

	if (!ground_truth_cube_tree_)
	{
		boost::property_tree::ptree ground_truth_cube_tree;

		try
		{
			boost::property_tree::read_json(dataset_path_ + parameter_.ground_truth_cube_file_name,
					ground_truth_cube_tree);
		} catch (boost::exception const &e)
		{
			std::cout << "[WARN]: Failed to load cube ground truth file from \"" << dataset_path_
					<< "\"." << std::endl;
			return nullptr;
		}

		ground_truth_cube_tree_ = boost::optional<boost::property_tree::ptree>(
				ground_truth_cube_tree);
	}

	std::shared_ptr<BoundingBox::Frame<BoundingBox::Cube>> ground_truth_frame_cube;
	boost::property_tree::ptree &ground_truth_cube_tree = *ground_truth_cube_tree_;

	auto ground_truth_frame_cube_tree_optioal = ground_truth_cube_tree.get_child_optional(
			boost::property_tree::ptree::path_type(frame_name_range, '/'));

	if (!ground_truth_frame_cube_tree_optioal)
	{
		std::cout << "[WARN]: Missing cube ground truth in frame \"" << frame_name_range << "\"."
				<< std::endl;
		return ground_truth_frame_cube;
	}

	ground_truth_frame_cube = std::make_shared<BoundingBox::Frame<BoundingBox::Cube>>();
	auto ground_truth_frame_cube_tree = *ground_truth_frame_cube_tree_optioal;

	for (const auto &ground_truth_cube_array_pair : ground_truth_frame_cube_tree)
	{
		std::vector<std::string> ground_truth_cube_values;
		Eigen::Vector3f center;
		Eigen::Vector3f extent;
		float rotation;
		std::string id;

		for (const auto &ground_truth_cube_value_pair : ground_truth_cube_array_pair.second)
		{
			ground_truth_cube_values.push_back(
					ground_truth_cube_value_pair.second.get_value<std::string>());
		}

		center << std::stof(ground_truth_cube_values[0]), std::stof(ground_truth_cube_values[1]), std::stof(
				ground_truth_cube_values[2]);
		extent << std::stof(ground_truth_cube_values[3]), std::stof(ground_truth_cube_values[4]), std::stof(
				ground_truth_cube_values[5]);
		rotation = std::stof(ground_truth_cube_values[6]);
		id = ground_truth_cube_values[7];

		ground_truth_frame_cube->push_back(std::make_tuple(center, extent, rotation, -1, id));
	}

	return ground_truth_frame_cube;
}

std::shared_ptr<BoundingBox::Frame<BoundingBox::Flat>>
DepthClustering::getGroundTruthFrameFlat(const std::string& frame_path_name_range)
{
	if (frame_path_name_range == "")
	{
		std::cout << "[WARN]: Invalid range frame path and name." << std::endl;
		return nullptr;
	}

	std::string frame_name_range = "";
	std::stringstream ss(frame_path_name_range);

	while (std::getline(ss, frame_name_range, '/'))
	{
	}

	if (frame_name_range == "")
	{
		std::cout << "[WARN]: Invalid range frame name." << std::endl;
		return nullptr;
	}

	if (!ground_truth_flat_tree_)
	{
		boost::property_tree::ptree ground_truth_flat_tree;

		try
		{
			boost::property_tree::read_json(dataset_path_ + parameter_.ground_truth_flat_file_name,
					ground_truth_flat_tree);
		} catch (boost::exception const &e)
		{
			std::cout << "[WARN]: Failed to load flat ground truth file from \"" << dataset_path_
					<< "\"." << std::endl;
			return nullptr;
		}

		ground_truth_flat_tree_ = boost::optional<boost::property_tree::ptree>(
				ground_truth_flat_tree);
	}

	std::shared_ptr<BoundingBox::Frame<BoundingBox::Flat>> ground_truth_frame_flat;
	boost::property_tree::ptree &ground_truth_flat_tree = *ground_truth_flat_tree_;

	auto ground_truth_frame_flat_tree_optioal = ground_truth_flat_tree.get_child_optional(
			boost::property_tree::ptree::path_type(frame_name_range, '/'));

	if (!ground_truth_frame_flat_tree_optioal)
	{
		std::cout << "[WARN]: Missing flat ground truth in frame \"" << frame_name_range << "\"."
				<< std::endl;
		return ground_truth_frame_flat;
	}

	ground_truth_frame_flat = std::make_shared<BoundingBox::Frame<BoundingBox::Flat>>();
	auto ground_truth_frame_flat_tree = *ground_truth_frame_flat_tree_optioal;

	for (const auto &ground_truth_flat_array_pair : ground_truth_frame_flat_tree)
	{
		std::vector<std::string> ground_truth_flat_values;
		Eigen::Vector2i corner_upper_left;
		Eigen::Vector2i corner_lower_right;
		float depth;
		std::string id;

		for (const auto &ground_truth_flat_value_pair : ground_truth_flat_array_pair.second)
		{
			ground_truth_flat_values.push_back(
					ground_truth_flat_value_pair.second.get_value<std::string>());
		}

		corner_upper_left << std::stoi(ground_truth_flat_values[0]), std::stoi(
				ground_truth_flat_values[1]);
		corner_lower_right << std::stoi(ground_truth_flat_values[2]), std::stoi(
				ground_truth_flat_values[3]);
		depth = std::stof(ground_truth_flat_values[4]);
		id = ground_truth_flat_values[5];

		ground_truth_frame_flat->push_back(
				std::make_tuple(corner_upper_left, corner_lower_right, depth, -1, id));
	}

	return ground_truth_frame_flat;
}

std::shared_ptr<ImageBasedClusterer<LinearImageLabeler<>>>
DepthClustering::getClusterer() const
{
	return clusterer_;
}

std::shared_ptr<FolderReader>
DepthClustering::getFolderReaderCamera()
{
	if (!folder_reader_camera_)
	{
		std::string dataset_path_camera = dataset_path_ + "frames_camera/";
		folder_reader_camera_ = std::make_shared<FolderReader>(dataset_path_camera, ".png",
				FolderReader::Order::SORTED);
	}

	return folder_reader_camera_;
}

std::shared_ptr<FolderReader>
DepthClustering::getFolderReaderRange() const
{
	return folder_reader_range_;
}

std::shared_ptr<FolderReader>
DepthClustering::getFolderReaderIntensity() const
{
	return folder_reader_intensity_;
}

std::shared_ptr<FolderReader>
DepthClustering::getFolderReaderElongation() const
{
	return folder_reader_elongation_;
}

std::shared_ptr<DepthGroundRemover>
DepthClustering::getDepthGroundRemover() const
{
	return depth_ground_remover_;
}

void
DepthClustering::setParameter(const DepthClusteringParameter& parameter)
{
	parameter_ = parameter;

	if (parameter_.use_camera_fov)
	{
		auto projection_parameter_lidar_raw = parameter_projection_lidar_->getProjectionParamsRaw();

		projection_parameter_lidar_raw->updateHorizontalAngles(
				parameter_projection_camera_.field_of_view_angle_start,
				parameter_projection_camera_.field_of_view_angle_end);

		parameter_projection_lidar_ = ProjectionParams::FromBeamInclinations(
				projection_parameter_lidar_raw->horizontal_steps_current,
				projection_parameter_lidar_raw->beams,
				projection_parameter_lidar_raw->horizontal_angle_start,
				projection_parameter_lidar_raw->horizontal_angle_end,
				projection_parameter_lidar_raw->beam_inclinations);

		parameter_projection_lidar_->setProjectionParamsRaw(projection_parameter_lidar_raw);
	}
	else
	{
		parameter_projection_lidar_ = parameter_factory_->getLidarProjectionParameter();

		if (parameter_factory_global_)
		{
			parameter_factory_global_->setGlobalLidarProjectionParameter(
					parameter_projection_lidar_);
		}
	}

	auto logger_parameter = parameter_factory_->getLoggerParameter();

	if (parameter_factory_global_)
	{
		parameter_factory_global_->setGlobalLoggerParameter(logger_parameter);
	}

	depth_ground_remover_ = std::make_shared<DepthGroundRemover>(*parameter_projection_lidar_,
			parameter_.angle_ground_removal, parameter_.size_smooth_window);
	score_ = std::make_shared<Score>(parameter_.score_type_point, parameter_.score_type_cluster,
			parameter_.score_type_frame);
	bounding_box_ = std::make_shared<BoundingBox>(score_, parameter_, parameter_projection_camera_);

	Radians clustering_threshold;

	switch (parameter_.difference_type)
	{
	case DiffFactory::DiffType::ANGLES:
	{
		clustering_threshold = parameter_.angle_clustering;
		break;
	}
	case DiffFactory::DiffType::ANGLES_PRECOMPUTED:
	{
		clustering_threshold = parameter_.angle_clustering;
		break;
	}
	case DiffFactory::DiffType::LINE_DIST:
	{
		clustering_threshold = Radians::FromRadians(parameter_.distance_clustering);
		break;
	}
	case DiffFactory::DiffType::LINE_DIST_PRECOMPUTED:
	{
		clustering_threshold = Radians::FromRadians(parameter_.distance_clustering);
		break;
	}
	case DiffFactory::DiffType::SIMPLE:
	{
		clustering_threshold = Radians::FromRadians(parameter_.distance_clustering);
		break;
	}
	default:
	{
		clustering_threshold = parameter_.angle_clustering;
		break;
	}
	}

	clusterer_ = std::make_shared<ImageBasedClusterer<LinearImageLabeler<>>>(clustering_threshold,
			parameter_.size_cluster_min, parameter_.size_cluster_max, parameter_.score_clustering);

	clusterer_->SetDiffType(parameter_.difference_type);
	logger_->setBoundingBox(bounding_box_);

	depth_ground_remover_->DepthGroundRemover::SenderTCloud::AddClient(clusterer_.get()); // @suppress("Method cannot be resolved")
	clusterer_->AddClient(bounding_box_.get());
}

void
DepthClustering::disableConsoleLogging()
{
	std::cout.rdbuf(NULL);
	stdout = freopen("/dev/null", "w", stdout);
}

bool
DepthClustering::initializeForApollo(const std::string& file_path_name_config,
		const std::string& log_path)
{
	parameter_factory_ = std::make_shared<ParameterFactory>(file_path_name_config);
	parameter_ = parameter_factory_->getDepthClusteringParameter();
	parameter_projection_lidar_ = parameter_factory_->getLidarProjectionParameter();
	auto logger_parameter = parameter_factory_->getLoggerParameter();

	logger_parameter.log_path = log_path;

	depth_ground_remover_ = std::make_shared<DepthGroundRemover>(*parameter_projection_lidar_,
			parameter_.angle_ground_removal, parameter_.size_smooth_window);
	score_ = std::make_shared<Score>(parameter_.score_type_point, parameter_.score_type_cluster,
			parameter_.score_type_frame);
	bounding_box_ = std::make_shared<BoundingBox>(score_, parameter_);
	logger_ = std::make_shared<Logger>(logger_parameter);

	ground_truth_cube_tree_ = boost::none;
	ground_truth_flat_tree_ = boost::none;

	Radians clustering_threshold;

	switch (parameter_.difference_type)
	{
	case DiffFactory::DiffType::ANGLES:
	{
		clustering_threshold = parameter_.angle_clustering;
		break;
	}
	case DiffFactory::DiffType::ANGLES_PRECOMPUTED:
	{
		clustering_threshold = parameter_.angle_clustering;
		break;
	}
	case DiffFactory::DiffType::LINE_DIST:
	{
		clustering_threshold = Radians::FromRadians(parameter_.distance_clustering);
		break;
	}
	case DiffFactory::DiffType::LINE_DIST_PRECOMPUTED:
	{
		clustering_threshold = Radians::FromRadians(parameter_.distance_clustering);
		break;
	}
	case DiffFactory::DiffType::SIMPLE:
	{
		clustering_threshold = Radians::FromRadians(parameter_.distance_clustering);
		break;
	}
	default:
	{
		clustering_threshold = parameter_.angle_clustering;
		break;
	}
	}

	clusterer_ = std::make_shared<ImageBasedClusterer<LinearImageLabeler<>>>(clustering_threshold,
			parameter_.size_cluster_min, parameter_.size_cluster_max, parameter_.score_clustering);

	clusterer_->SetDiffType(parameter_.difference_type);
	logger_->setBoundingBox(bounding_box_);

	depth_ground_remover_->DepthGroundRemover::SenderTCloud::AddClient(clusterer_.get()); // @suppress("Method cannot be resolved")
	clusterer_->AddClient(bounding_box_.get());

	return true;
}

bool
DepthClustering::initializeForDataset(const std::string& dataset_path,
		const std::string& file_path_name_config_global, std::string directory_name_lidar_return)
{
	dataset_path_ = dataset_path;

	if (dataset_path_ != "" && dataset_path_[dataset_path_.size() - 1] != '/')
	{
		dataset_path_ += "/";
	}

	if (directory_name_lidar_return == "")
	{
		directory_name_lidar_return = "first_return";
	}
	else if (directory_name_lidar_return != "first_return"
			&& directory_name_lidar_return != "second_return"
			&& directory_name_lidar_return != "custom_return")
	{
		std::cout << "[WARN]: Invalid lidar return directory: \"" << directory_name_lidar_return
				<< "\"." << std::endl;
		directory_name_lidar_return = "first_return";
	}

	std::cout << "[INFO]: Using lidar return directory \"" << directory_name_lidar_return << "\"."
			<< std::endl;

	parameter_factory_ = std::make_shared<ParameterFactory>(
			dataset_path_ + "depth_clustering_config.json");
	parameter_ = parameter_factory_->getDepthClusteringParameter();
	parameter_projection_lidar_ = parameter_factory_->getLidarProjectionParameter();
	parameter_projection_camera_ = parameter_factory_->getCameraProjectionParameter();
	auto logger_parameter = parameter_factory_->getLoggerParameter();

	if (file_path_name_config_global != "")
	{
		parameter_factory_global_ = std::make_shared<ParameterFactory>(
				file_path_name_config_global);
		parameter_factory_global_->setGlobalDepthClusteringParameter(parameter_);
		parameter_factory_global_->setGlobalLidarProjectionParameter(parameter_projection_lidar_);
		parameter_factory_global_->setGlobalCameraProjectionParameter(parameter_projection_camera_);
		parameter_factory_global_->setGlobalLoggerParameter(logger_parameter);
	}

	if (parameter_.use_camera_fov)
	{
		auto projection_parameter_lidar_raw = parameter_projection_lidar_->getProjectionParamsRaw();

		if (!projection_parameter_lidar_raw)
		{
			std::cout << "[ERROR]: Raw lidar projection parameter missing." << std::endl;
			return false;
		}

		projection_parameter_lidar_raw->updateHorizontalAngles(
				parameter_projection_camera_.field_of_view_angle_start,
				parameter_projection_camera_.field_of_view_angle_end);

		parameter_projection_lidar_ = ProjectionParams::FromBeamInclinations(
				projection_parameter_lidar_raw->horizontal_steps_current,
				projection_parameter_lidar_raw->beams,
				projection_parameter_lidar_raw->horizontal_angle_start,
				projection_parameter_lidar_raw->horizontal_angle_end,
				projection_parameter_lidar_raw->beam_inclinations);

		parameter_projection_lidar_->setProjectionParamsRaw(projection_parameter_lidar_raw);
	}

	// Remove extrinsic translation vector since bounding boxes are already in camera frame
	if (parameter_projection_camera_.extrinsic.size() >= 12)
	{
		parameter_projection_camera_.extrinsic[3] = 0;
		parameter_projection_camera_.extrinsic[7] = 0;
		parameter_projection_camera_.extrinsic[11] = 0;
	}

	logger_parameter.log_path = dataset_path_ + "frames_lidar/" + directory_name_lidar_return;

	std::string dataset_path_range = dataset_path_ + "frames_lidar/" + directory_name_lidar_return
			+ "/range/";
	std::string dataset_path_intensity = dataset_path_ + "frames_lidar/"
			+ directory_name_lidar_return + "/intensity/";
	std::string dataset_path_elongation = dataset_path_ + "frames_lidar/"
			+ directory_name_lidar_return + "/elongation/";

	folder_reader_range_ = std::make_shared<FolderReader>(dataset_path_range,
			parameter_.dataset_file_type, FolderReader::Order::SORTED);
	folder_reader_intensity_ = std::make_shared<FolderReader>(dataset_path_intensity,
			parameter_.dataset_file_type, FolderReader::Order::SORTED);
	folder_reader_elongation_ = std::make_shared<FolderReader>(dataset_path_elongation,
			parameter_.dataset_file_type, FolderReader::Order::SORTED);
	folder_reader_camera_ = nullptr;

	depth_ground_remover_ = std::make_shared<DepthGroundRemover>(*parameter_projection_lidar_,
			parameter_.angle_ground_removal, parameter_.size_smooth_window);
	score_ = std::make_shared<Score>(parameter_.score_type_point, parameter_.score_type_cluster,
			parameter_.score_type_frame);
	bounding_box_ = std::make_shared<BoundingBox>(score_, parameter_, parameter_projection_camera_);
	logger_ = std::make_shared<Logger>(logger_parameter);

	ground_truth_cube_tree_ = boost::none;
	ground_truth_flat_tree_ = boost::none;

	Radians clustering_threshold;

	switch (parameter_.difference_type)
	{
	case DiffFactory::DiffType::ANGLES:
	{
		clustering_threshold = parameter_.angle_clustering;
		break;
	}
	case DiffFactory::DiffType::ANGLES_PRECOMPUTED:
	{
		clustering_threshold = parameter_.angle_clustering;
		break;
	}
	case DiffFactory::DiffType::LINE_DIST:
	{
		clustering_threshold = Radians::FromRadians(parameter_.distance_clustering);
		break;
	}
	case DiffFactory::DiffType::LINE_DIST_PRECOMPUTED:
	{
		clustering_threshold = Radians::FromRadians(parameter_.distance_clustering);
		break;
	}
	case DiffFactory::DiffType::SIMPLE:
	{
		clustering_threshold = Radians::FromRadians(parameter_.distance_clustering);
		break;
	}
	default:
	{
		clustering_threshold = parameter_.angle_clustering;
		break;
	}
	}

	clusterer_ = std::make_shared<ImageBasedClusterer<LinearImageLabeler<>>>(clustering_threshold,
			parameter_.size_cluster_min, parameter_.size_cluster_max, parameter_.score_clustering);

	clusterer_->SetDiffType(parameter_.difference_type);
	logger_->setBoundingBox(bounding_box_);

	depth_ground_remover_->DepthGroundRemover::SenderTCloud::AddClient(clusterer_.get()); // @suppress("Method cannot be resolved")
	clusterer_->AddClient(bounding_box_.get());

	return true;
}

const std::string
DepthClustering::processOneFrameForApollo(const std::string& frame_name,
		const std::vector<Eigen::Vector3f>& point_cloud)
{
	Timer timer;

	std::cout << "[INFO]: Processing \"" << frame_name << "\"." << std::endl;

	cloud_ = Cloud::Ptr(new Cloud);

	for (const auto &point_eigen : point_cloud)
	{
		RichPoint point_rich;

		point_rich.x() = point_eigen.x();
		point_rich.y() = point_eigen.y();
		point_rich.z() = point_eigen.z();

		cloud_->push_back(point_rich);
	}

	cloud_->InitProjection(*parameter_projection_lidar_);

	bounding_box_->clearFrames();

	std::cout << "[INFO]: Preprocessed: " << timer.measure() << " us." << std::endl;

	depth_ground_remover_->OnNewObjectReceived(*cloud_, 0);

	std::cout << "[INFO]: Clustered: " << timer.measure() << " us." << std::endl;

	logger_->logBoundingBoxFrame(frame_name, parameter_.bounding_box_type);

	std::cout << "[INFO]: Logged: " << timer.measure() << " us." << std::endl;

	return frame_name;
}

const std::string
DepthClustering::processOneFrameForDataset(const std::string& frame_path_name_range,
		const std::string& frame_path_name_intensity, const std::string& frame_path_name_elongation)
{
	if (frame_path_name_range == "")
	{
		std::cerr << "[ERROR]: Invalid range frame path and name." << std::endl;
		return "";
	}

	Timer timer;
	std::string frame_name_range = "";
	std::string frame_name_intensity = "";
	std::string frame_name_elongation = "";
	std::stringstream ss_range(frame_path_name_range);
	std::stringstream ss_intensity(frame_path_name_intensity);
	std::stringstream ss_elongation(frame_path_name_elongation);

	while (std::getline(ss_range, frame_name_range, '/'))
	{
	}

	while (std::getline(ss_intensity, frame_name_intensity, '/'))
	{
	}

	while (std::getline(ss_elongation, frame_name_elongation, '/'))
	{
	}

	if (frame_name_range == "")
	{
		std::cerr << "[ERROR]: Invalid range frame name." << std::endl;
		return "";
	}

	std::cout << "[INFO]: Processing \"" << frame_name_range << "\"." << std::endl;

	if (parameter_.dataset_file_type == ".png")
	{
		image_range_ = MatFromPNGRange(frame_path_name_range, parameter_projection_lidar_);
		image_intensity_ = MatFromPNGIntensity(frame_path_name_intensity,
				parameter_projection_lidar_);
		image_elongation_ = MatFromPNGElongation(frame_path_name_elongation,
				parameter_projection_lidar_);
	}
	else if (parameter_.dataset_file_type == ".tiff")
	{
		image_range_ = MatFromTIFFRange(frame_path_name_range, parameter_projection_lidar_);
		image_intensity_ = MatFromTIFFIntensity(frame_path_name_intensity,
				parameter_projection_lidar_);
		image_elongation_ = MatFromTIFFElongation(frame_path_name_elongation,
				parameter_projection_lidar_);
	}
	else
	{
		std::cerr << "[ERROR]: Unknown dataset file type." << std::endl;
		return "";
	}

	cloud_ = Cloud::FromImage(image_range_, image_intensity_, image_elongation_, score_,
			*parameter_projection_lidar_);

	bounding_box_->clearFrames();

	std::cout << "[INFO]: Preprocessed: " << timer.measure() << " us." << std::endl;

	depth_ground_remover_->OnNewObjectReceived(*cloud_, 0);

	std::cout << "[INFO]: Clustered: " << timer.measure() << " us." << std::endl;

	bounding_box_->produceFrameFlat();

	std::cout << "[INFO]: Bounding box projected: " << timer.measure() << " us." << std::endl;

	logger_->logBoundingBoxFrame(frame_name_range, parameter_.bounding_box_type);
	logger_->logBoundingBoxFrameFlat(frame_name_range);

	std::cout << "[INFO]: Logged: " << timer.measure() << " us." << std::endl;

	return frame_name_range;
}

void
DepthClustering::writeLogForApollo()
{
	std::cout << std::endl;
	logger_->writeBoundingBoxLog(parameter_.bounding_box_type);
}

void
DepthClustering::writeLogForDataset()
{
	std::cout << std::endl;
	logger_->writeBoundingBoxLog(parameter_.bounding_box_type);
	logger_->writeBoundingBoxLog(BoundingBox::Type::Flat);
}
} // namespace depth_clustering
