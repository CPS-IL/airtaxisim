/*
 * camera_projection.cpp
 *
 *  Created on: Sep 10, 2020
 *      Author: simonyu
 */

#include "post_processing/camera_projection.h"

namespace depth_clustering
{
CameraProjection::CameraProjection() :
		CameraProjection(CameraProjectionParameter())
{
}

CameraProjection::CameraProjection(const CameraProjectionParameter& parameter) :
		parameter_(parameter)
{
}

void
CameraProjection::setFrames(
		std::shared_ptr<BoundingBox::Frame<BoundingBox::Cube>> bounding_box_frame_cube,
		std::shared_ptr<BoundingBox::Frame<BoundingBox::Polygon>> bounding_box_frame_polygon,
		std::shared_ptr<BoundingBox::Frame<BoundingBox::Flat>> bounding_box_frame_flat)
{
	bounding_box_frame_cube_ = bounding_box_frame_cube;
	bounding_box_frame_polygon_ = bounding_box_frame_polygon;
	bounding_box_frame_flat_ = bounding_box_frame_flat;
}

void
CameraProjection::projectFromBoundingBoxFrame(const BoundingBox::Type& bounding_box_type)
{
	switch (bounding_box_type)
	{
	case BoundingBox::Type::Cube:
	{
		projectFromBoundingBoxFrameCube();
		break;
	}
	case BoundingBox::Type::Polygon:
	{
		projectFromBoundingBoxFramePolygon();
		break;
	}
	default:
	{
		projectFromBoundingBoxFrameCube();
		break;
	}
	}
}

std::vector<Eigen::Vector3f>
CameraProjection::getBoundingBoxCornersCube(const BoundingBox::Cube& bounding_box)
{
	std::vector<Eigen::Vector3f> bounding_box_corners;
	Eigen::Matrix4f bounding_box_to_world; // Bounding box transformation matrix
	auto bounding_box_unit_center = Eigen::Vector3f(0, 0, 0);
	auto bounding_box_unit_extent = Eigen::Vector3f(1, 1, 1);
	auto bounding_box_center = std::get<0>(bounding_box);
	auto bounding_box_extent = std::get<1>(bounding_box);
	auto bounding_box_rotation = std::get<2>(bounding_box);

	// Construct bounding box transformation matrix
	bounding_box_to_world << bounding_box_extent.x() * cos(bounding_box_rotation), -1
			* bounding_box_extent.y() * sin(bounding_box_rotation), 0, bounding_box_center.x(), bounding_box_extent.x()
			* sin(bounding_box_rotation), bounding_box_extent.y() * cos(bounding_box_rotation), 0, bounding_box_center.y(), 0, 0, bounding_box_extent.z(), bounding_box_center.z(), 0, 0, 0, 1;

	bounding_box_corners.push_back(
			Eigen::Vector3f(bounding_box_unit_center.x() - bounding_box_unit_extent.x() / 2,
					bounding_box_unit_center.y() - bounding_box_unit_extent.y() / 2,
					bounding_box_unit_center.z() - bounding_box_unit_extent.z() / 2));

	bounding_box_corners.push_back(
			Eigen::Vector3f(bounding_box_unit_center.x() - bounding_box_unit_extent.x() / 2,
					bounding_box_unit_center.y() - bounding_box_unit_extent.y() / 2,
					bounding_box_unit_center.z() + bounding_box_unit_extent.z() / 2));

	bounding_box_corners.push_back(
			Eigen::Vector3f(bounding_box_unit_center.x() - bounding_box_unit_extent.x() / 2,
					bounding_box_unit_center.y() + bounding_box_unit_extent.y() / 2,
					bounding_box_unit_center.z() - bounding_box_unit_extent.z() / 2));

	bounding_box_corners.push_back(
			Eigen::Vector3f(bounding_box_unit_center.x() - bounding_box_unit_extent.x() / 2,
					bounding_box_unit_center.y() + bounding_box_unit_extent.y() / 2,
					bounding_box_unit_center.z() + bounding_box_unit_extent.z() / 2));

	bounding_box_corners.push_back(
			Eigen::Vector3f(bounding_box_unit_center.x() + bounding_box_unit_extent.x() / 2,
					bounding_box_unit_center.y() - bounding_box_unit_extent.y() / 2,
					bounding_box_unit_center.z() - bounding_box_unit_extent.z() / 2));

	bounding_box_corners.push_back(
			Eigen::Vector3f(bounding_box_unit_center.x() + bounding_box_unit_extent.x() / 2,
					bounding_box_unit_center.y() - bounding_box_unit_extent.y() / 2,
					bounding_box_unit_center.z() + bounding_box_unit_extent.z() / 2));

	bounding_box_corners.push_back(
			Eigen::Vector3f(bounding_box_unit_center.x() + bounding_box_unit_extent.x() / 2,
					bounding_box_unit_center.y() + bounding_box_unit_extent.y() / 2,
					bounding_box_unit_center.z() - bounding_box_unit_extent.z() / 2));

	bounding_box_corners.push_back(
			Eigen::Vector3f(bounding_box_unit_center.x() + bounding_box_unit_extent.x() / 2,
					bounding_box_unit_center.y() + bounding_box_unit_extent.y() / 2,
					bounding_box_unit_center.z() + bounding_box_unit_extent.z() / 2));

	// Transform unit bounding box corners into world frame
	for (auto &bounding_box_corner : bounding_box_corners)
	{
		auto bounding_box_corner_extended = Eigen::Vector4f(bounding_box_corner.x(),
				bounding_box_corner.y(), bounding_box_corner.z(), 1);

		bounding_box_corner_extended = bounding_box_to_world * bounding_box_corner_extended;
		bounding_box_corner.x() = bounding_box_corner_extended.x();
		bounding_box_corner.y() = bounding_box_corner_extended.y();
		bounding_box_corner.z() = bounding_box_corner_extended.z();
	}

	return bounding_box_corners;
}

std::vector<Eigen::Vector3f>
CameraProjection::getBoundingBoxCornersPolygon(const BoundingBox::Polygon& bounding_box)
{
	std::vector<Eigen::Vector3f> bounding_box_corners;
	const auto bounding_box_hull = std::get<0>(bounding_box);
	const auto bounding_box_diff_z = std::get<1>(bounding_box);

	for (const auto &bounding_box_corner_bottom : bounding_box_hull)
	{
		auto bounding_box_corner_top = bounding_box_corner_bottom;

		bounding_box_corner_top.z() += bounding_box_diff_z;
		bounding_box_corners.push_back(bounding_box_corner_bottom);
		bounding_box_corners.push_back(bounding_box_corner_top);
	}

	return bounding_box_corners;
}

float
CameraProjection::getBoundingBoxDepth(const std::vector<Eigen::Vector3f>& bounding_box_corners)
{
	float depth = bounding_box_corners[0].norm();

	for (const auto &bounding_box_corner : bounding_box_corners)
	{
		float depth_current = bounding_box_corner.norm();

		if (depth_current < depth)
		{
			depth = depth_current;
		}
	}

	return depth;
}

float
CameraProjection::getBoundingBoxBoundTop(const std::vector<Eigen::Vector3f>& bounding_box_corners)
{
	float bound_top = bounding_box_corners[0].z();

	for (const auto &bounding_box_corner : bounding_box_corners)
	{
		float bound_top_current = bounding_box_corner.z();

		if (bound_top_current > bound_top)
		{
			bound_top = bound_top_current;
		}
	}

	return bound_top;
}

float
CameraProjection::getBoundingBoxBoundLeft(const std::vector<Eigen::Vector3f>& bounding_box_corners)
{
	float bound_left = bounding_box_corners[0].y();

	for (const auto &bounding_box_corner : bounding_box_corners)
	{
		float bound_left_current = bounding_box_corner.y();

		if (bound_left_current < bound_left)
		{
			bound_left = bound_left_current;
		}
	}

	return bound_left;
}

float
CameraProjection::getBoundingBoxBoundRight(const std::vector<Eigen::Vector3f>& bounding_box_corners)
{
	float bound_right = bounding_box_corners[0].y();

	for (const auto &bounding_box_corner : bounding_box_corners)
	{
		float bound_right_current = bounding_box_corner.y();

		if (bound_right_current > bound_right)
		{
			bound_right = bound_right_current;
		}
	}

	return bound_right;
}

std::shared_ptr<BoundingBox::Flat>
CameraProjection::getBoundingBoxFlat(
		const std::vector<Eigen::Vector2i>& bounding_box_corners_projected)
{
	Eigen::Vector2i bounding_box_flat_corner_upper_left;
	Eigen::Vector2i bounding_box_flat_corner_lower_right;
	double bounding_box_flat_area_original;
	double bounding_box_flat_area;
	double area_percentage;
	double truncation;

	bounding_box_flat_corner_upper_left.x() = std::numeric_limits<int>::max();
	bounding_box_flat_corner_lower_right.x() = std::numeric_limits<int>::min();
	bounding_box_flat_corner_upper_left.y() = std::numeric_limits<int>::max();
	bounding_box_flat_corner_lower_right.y() = std::numeric_limits<int>::min();

	// Find flat bounding box boundary
	for (const auto &bounding_box_corner_projected : bounding_box_corners_projected)
	{
		if (bounding_box_corner_projected.x() < bounding_box_flat_corner_upper_left.x())
		{
			bounding_box_flat_corner_upper_left.x() = bounding_box_corner_projected.x();
		}

		if (bounding_box_corner_projected.x() > bounding_box_flat_corner_lower_right.x())
		{
			bounding_box_flat_corner_lower_right.x() = bounding_box_corner_projected.x();
		}

		if (bounding_box_corner_projected.y() < bounding_box_flat_corner_upper_left.y())
		{
			bounding_box_flat_corner_upper_left.y() = bounding_box_corner_projected.y();
		}

		if (bounding_box_corner_projected.y() > bounding_box_flat_corner_lower_right.y())
		{
			bounding_box_flat_corner_lower_right.y() = bounding_box_corner_projected.y();
		}
	}

	// Calculate original dimension
	double bounding_box_flat_width_original = bounding_box_flat_corner_lower_right.x()
			- bounding_box_flat_corner_upper_left.x();
	double bounding_box_flat_height_original = bounding_box_flat_corner_lower_right.y()
			- bounding_box_flat_corner_upper_left.y();
	bounding_box_flat_area_original = bounding_box_flat_width_original
			* bounding_box_flat_height_original;

	// Exclude bounding box with no area
	if (bounding_box_flat_area_original == 0)
	{
		return nullptr;
	}

	// Enforce image boundary
	bounding_box_flat_corner_upper_left.x() = std::min<int>(
			std::max<int>(0, bounding_box_flat_corner_upper_left.x()), parameter_.width - 1);
	bounding_box_flat_corner_lower_right.x() = std::min<int>(
			std::max<int>(0, bounding_box_flat_corner_lower_right.x()), parameter_.width - 1);
	bounding_box_flat_corner_upper_left.y() = std::min<int>(
			std::max<int>(0, bounding_box_flat_corner_upper_left.y()), parameter_.height - 1);
	bounding_box_flat_corner_lower_right.y() = std::min<int>(
			std::max<int>(0, bounding_box_flat_corner_lower_right.y()), parameter_.height - 1);

	// Calculate dimension
	double bounding_box_flat_width = bounding_box_flat_corner_lower_right.x()
			- bounding_box_flat_corner_upper_left.x();
	double bounding_box_flat_height = bounding_box_flat_corner_lower_right.y()
			- bounding_box_flat_corner_upper_left.y();
	bounding_box_flat_area = bounding_box_flat_width * bounding_box_flat_height;

	// Exclude bounding box with no area
	if (bounding_box_flat_area == 0)
	{
		return nullptr;
	}

	// Calculate truncation
	area_percentage = bounding_box_flat_area / bounding_box_flat_area_original;
	truncation = 1 - area_percentage;

	// If truncated more than threshold
	if (truncation >= parameter_.threshold_truncation)
	{
		return nullptr;
	}

	return std::make_shared<BoundingBox::Flat>(
			std::make_tuple(bounding_box_flat_corner_upper_left,
					bounding_box_flat_corner_lower_right, 0, -1, ""));
}

void
CameraProjection::correctCameraDistortions(Eigen::Vector2f& point)
{
	if (!parameter_.correct_distortions)
	{
		return;
	}

	double k1 = parameter_.intrinsic[4];
	double k2 = parameter_.intrinsic[5];
	double p1 = parameter_.intrinsic[6];
	double p2 = parameter_.intrinsic[7];
	double k3 = parameter_.intrinsic[8];

	double r = std::sqrt(std::pow(point.x(), 2) + std::pow(point.y(), 2));
	double radial_correction_factor = 1 + k1 * std::pow(r, 2) + k2 * std::pow(r, 4)
			+ k3 * std::pow(r, 6);
	double tangential_correction_factor_x = 2 * p1 * point.x() * point.y()
			+ p2 * (std::pow(r, 2) + 2 * std::pow(point.x(), 2));
	double tangential_correction_factor_y = 2 * p2 * point.x() * point.y()
			+ p1 * (std::pow(r, 2) + 2 * std::pow(point.y(), 2));

	point.x() *= radial_correction_factor;
	point.y() *= radial_correction_factor;
	point.x() += tangential_correction_factor_x;
	point.y() += tangential_correction_factor_y;
}

bool
CameraProjection::filterBoundingBoxHeight(const float& bound_top)
{
	if (!parameter_.use_filter_height)
	{
		return true;
	}

	return bound_top < parameter_.threshold_filter_height;
}

bool
CameraProjection::filterBoundingBoxTunnel(const float& depth, const float& bound_left,
		const float& bound_right)
{
	if (!parameter_.use_filter_tunnel)
	{
		return true;
	}

	return (bound_right <= parameter_.threshold_filter_tunnel_right
			&& bound_left >= -parameter_.threshold_filter_tunnel_left
			&& depth <= parameter_.threshold_filter_tunnel_front);
}

void
CameraProjection::projectFromBoundingBoxFrameCube()
{
	if (!bounding_box_frame_cube_)
	{
		std::cout << "[ERROR]: Cube bounding box frame missing." << std::endl;
		return;
	}

	Eigen::Matrix4f camera_to_world; // Camera-to-world extrinsic transformation matrix, [R|t]
	Eigen::Matrix4f world_to_camera; // World-to-camera extrinsic transformation matrix, [R|t]
	Eigen::Matrix4f world_to_camera_axes; // World-to-camera extrinsic axes transformation matrix

	// Construct camera-to-world extrinsic transformation matrix
	for (size_t i = 0; i < parameter_.extrinsic.size(); i++)
	{
		camera_to_world(i / 4, i % 4) = parameter_.extrinsic[i];
	}

	// Construct world-to-camera extrinsic axes transformation matrix
	world_to_camera_axes << 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1;

	// Obtain world-to-camera extrinsic transformation matrix
	world_to_camera = world_to_camera_axes * camera_to_world.inverse();

	// Process all bounding boxes in the frame
	for (const auto &bounding_box : *bounding_box_frame_cube_)
	{
		std::vector<Eigen::Vector3f> bounding_box_corners_world = getBoundingBoxCornersCube(
				bounding_box);
		std::vector<Eigen::Vector2i> bounding_box_corners_projected;
		float bounding_box_depth = getBoundingBoxDepth(bounding_box_corners_world);
		bool bounding_box_invalid = false;

		// Exclude bounding box with invalid depth
		if (bounding_box_depth < 0)
		{
			continue;
		}

		const auto bounding_box_bound_top = getBoundingBoxBoundTop(bounding_box_corners_world);
		const auto bounding_box_bound_left = getBoundingBoxBoundLeft(bounding_box_corners_world);
		const auto bounding_box_bound_right = getBoundingBoxBoundRight(bounding_box_corners_world);

		// Exclude bounding box according to filters
		if (!filterBoundingBoxHeight(bounding_box_bound_top)
				|| !filterBoundingBoxTunnel(bounding_box_depth, bounding_box_bound_left,
						bounding_box_bound_right))
		{
			continue;
		}

		for (const auto &bounding_box_corner_world : bounding_box_corners_world)
		{
			// Project from world frame into camera frame
			Eigen::Vector4f bounding_box_corner_camera = world_to_camera
					* Eigen::Vector4f(bounding_box_corner_world(0), bounding_box_corner_world(1),
							bounding_box_corner_world(2), 1);

			// Exclude bounding box outside the camera image plane
			if (bounding_box_corner_camera.z() < 0)
			{
				bounding_box_invalid = true;
				break;
			}

			// Project from 3D camera frame into 2D
			Eigen::Vector2f bounding_box_corner_projected(
					bounding_box_corner_camera(0) / bounding_box_corner_camera(2),
					bounding_box_corner_camera(1) / bounding_box_corner_camera(2));

			// Correct camera distortions
			correctCameraDistortions(bounding_box_corner_projected);

			// Project from 2D camera frame onto image frame by applying camera intrinsic matrix
			bounding_box_corner_projected.x() = std::round(
					bounding_box_corner_projected.x() * parameter_.intrinsic[0]
							+ parameter_.intrinsic[2]);
			bounding_box_corner_projected.y() = std::round(
					bounding_box_corner_projected.y() * parameter_.intrinsic[1]
							+ parameter_.intrinsic[3]);

			// Store projected corner
			bounding_box_corners_projected.push_back(
					Eigen::Vector2i(bounding_box_corner_projected(0),
							bounding_box_corner_projected(1)));
		}

		// Exclude invalid bounding box
		if (bounding_box_invalid || bounding_box_corners_projected.empty())
		{
			continue;
		}

		// Obtain 2D flat bounding box
		auto bounding_box_flat = getBoundingBoxFlat(bounding_box_corners_projected);

		// Exclude invalid bounding box
		if (!bounding_box_flat)
		{
			continue;
		}

		std::get<2>(*bounding_box_flat) = bounding_box_depth;
		std::get<3>(*bounding_box_flat) = std::get<3>(bounding_box);
		std::get<4>(*bounding_box_flat) = std::get<4>(bounding_box);

		// Store 2D flat bounding box
		bounding_box_frame_flat_->push_back(*bounding_box_flat);
	}
}

void
CameraProjection::projectFromBoundingBoxFramePolygon()
{
	if (!bounding_box_frame_polygon_)
	{
		std::cout << "[ERROR]: Polygon bounding box frame missing." << std::endl;
		return;
	}

	Eigen::Matrix4f camera_to_world; // Camera-to-world extrinsic transformation matrix, [R|t]
	Eigen::Matrix4f world_to_camera; // World-to-camera extrinsic transformation matrix, [R|t]
	Eigen::Matrix4f world_to_camera_axes; // World-to-camera extrinsic axes transformation matrix

	// Construct camera-to-world extrinsic transformation matrix
	for (size_t i = 0; i < parameter_.extrinsic.size(); i++)
	{
		camera_to_world(i / 4, i % 4) = parameter_.extrinsic[i];
	}

	// Construct world-to-camera extrinsic axes transformation matrix
	world_to_camera_axes << 0, -1, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 1;

	// Obtain world-to-camera extrinsic transformation matrix
	world_to_camera = world_to_camera_axes * camera_to_world.inverse();

	// Process all bounding boxes in the frame
	for (const auto &bounding_box : *bounding_box_frame_polygon_)
	{
		std::vector<Eigen::Vector3f> bounding_box_corners_world = getBoundingBoxCornersPolygon(
				bounding_box);
		std::vector<Eigen::Vector2i> bounding_box_corners_projected;
		float bounding_box_depth = getBoundingBoxDepth(bounding_box_corners_world);
		bool bounding_box_invalid = false;

		// Exclude bounding box with invalid depth
		if (bounding_box_depth < 0)
		{
			continue;
		}

		const auto bounding_box_bound_top = getBoundingBoxBoundTop(bounding_box_corners_world);
		const auto bounding_box_bound_left = getBoundingBoxBoundLeft(bounding_box_corners_world);
		const auto bounding_box_bound_right = getBoundingBoxBoundRight(bounding_box_corners_world);

		// Exclude bounding box according to filters
		if (!filterBoundingBoxHeight(bounding_box_bound_top)
				|| !filterBoundingBoxTunnel(bounding_box_depth, bounding_box_bound_left,
						bounding_box_bound_right))
		{
			continue;
		}

		for (const auto &bounding_box_corner_world : bounding_box_corners_world)
		{
			// Project from world frame into camera frame
			Eigen::Vector4f bounding_box_corner_camera = world_to_camera
					* Eigen::Vector4f(bounding_box_corner_world(0), bounding_box_corner_world(1),
							bounding_box_corner_world(2), 1);

			// Exclude bounding box outside the camera image plane
			if (bounding_box_corner_camera.z() < 0)
			{
				bounding_box_invalid = true;
				break;
			}

			// Project from 3D camera frame into 2D
			Eigen::Vector2f bounding_box_corner_projected(
					bounding_box_corner_camera(0) / bounding_box_corner_camera(2),
					bounding_box_corner_camera(1) / bounding_box_corner_camera(2));

			// Correct camera distortions
			correctCameraDistortions(bounding_box_corner_projected);

			// Project from 2D camera frame onto image frame by applying camera intrinsic matrix
			bounding_box_corner_projected.x() = std::round(
					bounding_box_corner_projected.x() * parameter_.intrinsic[0]
							+ parameter_.intrinsic[2]);
			bounding_box_corner_projected.y() = std::round(
					bounding_box_corner_projected.y() * parameter_.intrinsic[1]
							+ parameter_.intrinsic[3]);

			// Store projected corner
			bounding_box_corners_projected.push_back(
					Eigen::Vector2i(bounding_box_corner_projected(0),
							bounding_box_corner_projected(1)));
		}

		// Exclude invalid bounding box
		if (bounding_box_invalid || bounding_box_corners_projected.empty())
		{
			continue;
		}

		// Obtain 2D flat bounding box
		auto bounding_box_flat = getBoundingBoxFlat(bounding_box_corners_projected);

		// Exclude invalid bounding box
		if (!bounding_box_flat)
		{
			continue;
		}

		std::get<2>(*bounding_box_flat) = bounding_box_depth;
		std::get<3>(*bounding_box_flat) = std::get<2>(bounding_box);
		std::get<4>(*bounding_box_flat) = std::get<3>(bounding_box);

		// Store 2D flat bounding box
		bounding_box_frame_flat_->push_back(*bounding_box_flat);
	}
}
} // namespace depth_clustering
