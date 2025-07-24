/*
 * bounding_box.cpp
 *
 *  Created on: Sep 9, 2020
 *      Author: simonyu
 */

#include "api/parameter.h"
#include "post_processing/bounding_box.h"
#include "post_processing/camera_projection.h"
#include "post_processing/score.h"

namespace depth_clustering
{
BoundingBox::BoundingBox() :
		BoundingBox(nullptr, DepthClusteringParameter())
{
}

BoundingBox::BoundingBox(const std::shared_ptr<Score> score,
		const DepthClusteringParameter& parameter) :
		frame_score_(-1)
{
	type_ = parameter.bounding_box_type;
	use_score_filter_ = parameter.use_score_filter;
	score_filter_threshold_ = parameter.score_filter_threshold;
	id_ = 0;
	frame_cluster_ = std::make_shared<Frame<Cluster>>();
	frame_cube_ = std::make_shared<Frame<Cube>>();
	frame_polygon_ = std::make_shared<Frame<Polygon>>();
	frame_flat_ = std::make_shared<Frame<Flat>>();
	score_ = score;
}

BoundingBox::BoundingBox(const std::shared_ptr<Score> score,
		const DepthClusteringParameter& parameter,
		const CameraProjectionParameter& camera_projection_parameter) :
		BoundingBox(score, parameter)
{
	camera_projection_ = std::make_shared<CameraProjection>(camera_projection_parameter);
	camera_projection_->setFrames(frame_cube_, frame_polygon_, frame_flat_);
}

std::shared_ptr<BoundingBox::Frame<BoundingBox::Cluster>>
BoundingBox::getFrameCluster() const
{
	return frame_cluster_;
}

std::shared_ptr<BoundingBox::Frame<BoundingBox::Cube>>
BoundingBox::getFrameCube() const
{
	return frame_cube_;
}

std::shared_ptr<BoundingBox::Frame<BoundingBox::Polygon>>
BoundingBox::getFramePolygon() const
{
	return frame_polygon_;
}

std::shared_ptr<BoundingBox::Frame<BoundingBox::Flat>>
BoundingBox::getFrameFlat() const
{
	return frame_flat_;
}

float
BoundingBox::getFrameScore() const
{
	return frame_score_;
}

void
BoundingBox::setFrameCube(std::shared_ptr<Frame<Cube>> frame_cube)
{
	frame_cube_ = frame_cube;
	camera_projection_->setFrames(frame_cube_, frame_polygon_, frame_flat_);
}

void
BoundingBox::setFramePolygon(std::shared_ptr<Frame<Polygon>> frame_polygon)
{
	frame_polygon_ = frame_polygon;
	camera_projection_->setFrames(frame_cube_, frame_polygon_, frame_flat_);
}

void
BoundingBox::setFrameFlat(std::shared_ptr<Frame<Flat>> frame_flat)
{
	frame_flat_ = frame_flat;
	camera_projection_->setFrames(frame_cube_, frame_polygon_, frame_flat_);
}

void
BoundingBox::clearFrames()
{
	frame_cluster_->clear();
	frame_cube_->clear();
	frame_polygon_->clear();
	frame_flat_->clear();
	frame_score_ = -1;
}

void
BoundingBox::produceFrameFlat()
{
	if (!camera_projection_)
	{
		std::cout << "[WARN]: Camera projection missing." << std::endl;
		return;
	}

	camera_projection_->projectFromBoundingBoxFrame(type_);
}

void
BoundingBox::OnNewObjectReceived(const std::unordered_map<uint16_t, Cloud>& clouds, int id)
{
	if (!score_)
	{
		std::cout << "[WARN]: Score missing." << std::endl;
	}

	for (const auto &cloud_labeled : clouds)
	{
		float cluster_score = -1;

		if (cloud_labeled.second.empty())
		{
			continue;
		}

		if (score_)
		{
			cluster_score = score_->calculateClusterScore(cloud_labeled.second);
		}

		if (use_score_filter_ && cluster_score < score_filter_threshold_)
		{
			continue;
		}

		BoundingBox::Cluster cluster = std::make_tuple(cloud_labeled.second, cluster_score,
				std::to_string(id_));

		frame_cluster_->push_back(cluster);

		switch (type_)
		{
		case Type::Cube:
		{
			CreateCubes(cluster);
			break;
		}
		case Type::Polygon:
		{
			CreatePolygons(cluster);
			break;
		}
		default:
		{
			CreateCubes(cluster);
			break;
		}
		}

		id_++;
	}

	if (score_)
	{
		frame_score_ = score_->calculateFrameScore(frame_cluster_);
	}
}

void
BoundingBox::CreateCubes(const Cluster& cluster)
{
	if (!frame_cube_)
	{
		std::cout << "[ERROR]: Cube frame missing." << std::endl;
		return;
	}

	const Cloud &cloud = std::get<0>(cluster);
	const float &score = std::get<1>(cluster);
	Eigen::Vector3f center = Eigen::Vector3f::Zero();
	Eigen::Vector3f extent = Eigen::Vector3f::Zero();
	Eigen::Vector3f max_point(std::numeric_limits<float>::lowest(),
			std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());
	Eigen::Vector3f min_point(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
			std::numeric_limits<float>::max());
	for (const auto &point : cloud.points())
	{
		min_point << std::min(min_point.x(), point.x()), std::min(min_point.y(), point.y()), std::min(
				min_point.z(), point.z());
		max_point << std::max(max_point.x(), point.x()), std::max(max_point.y(), point.y()), std::max(
				max_point.z(), point.z());
	}
	center = min_point + (max_point - min_point) / 2;
	if (min_point.x() < max_point.x())
	{
		extent = max_point - min_point;
	}

	frame_cube_->push_back(std::make_tuple(center, extent, 0, score, std::to_string(id_)));
}

void
BoundingBox::CreatePolygons(const Cluster& cluster)
{
	if (!frame_polygon_)
	{
		std::cout << "[ERROR]: Polygon frame missing." << std::endl;
		return;
	}

	const Cloud &cloud = std::get<0>(cluster);
	const float &score = std::get<1>(cluster);
	float min_z
	{ std::numeric_limits<float>::max() };
	float max_z
	{ std::numeric_limits<float>::lowest() };
	std::vector<cv::Point2f> cv_points;
	cv_points.reserve(cloud.size());
	std::vector<int> hull_indices;
	for (const auto &point : cloud.points())
	{
		cv_points.emplace_back(cv::Point2f
		{ point.x(), point.y() });
		min_z = std::min(min_z, point.z());
		max_z = std::max(max_z, point.z());
	}
	cv::convexHull(cv_points, hull_indices);
	AlignedEigenVectors hull;
	hull.reserve(cloud.size());
	for (int index : hull_indices)
	{
		const auto &cv_point = cv_points[index];
		hull.emplace_back(cv_point.x, cv_point.y, min_z);
	}
	const float height = max_z - min_z;

	frame_polygon_->push_back(std::make_tuple(hull, height, score, std::to_string(id_)));
}
} // namespace depth_clustering
