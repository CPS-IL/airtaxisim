// Copyright (C) 2017  I. Bogoslavskyi, C. Stachniss, University of Bonn

// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.

// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.

// You should have received a copy of the GNU General Public License along
// with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "post_processing/score.h"
#include "utils/cloud.h"

namespace depth_clustering
{
Cloud::Cloud(const Cloud& cloud) :
		_points
		{ cloud.points() }, _pose(cloud.pose()), _sensor_pose(cloud.sensor_pose())
{
	if (!cloud.projection_ptr())
	{
		// no need to copy projection, there is none yet
		return;
	}
	// projection is a polymorphic type, we use clone therefore
	auto ptr = cloud.projection_ptr()->Clone();
	_projection = ptr;
}

std::list<const RichPoint*>
Cloud::PointsProjectedToPixel(int row, int col) const
{
	std::list<const RichPoint*> point_list;
	if (!_projection)
	{
		return point_list;
	}
	for (const auto &index : _projection->at(row, col).points())
	{
		point_list.push_back(&_points[index]);
	}
	return point_list;
}

void
Cloud::TransformInPlace(const Pose& pose)
{
	for (auto &point : _points)
	{
		point = pose * point.AsEigenVector();
	}
	// the projection makes no sense anymore after the coords of points changed.
	this->_projection.reset();
}

Cloud::Ptr
Cloud::Transform(const Pose& pose) const
{
	Cloud cloud_copy(*this);
	cloud_copy.TransformInPlace(pose);
	return std::make_shared<Cloud>(cloud_copy);
}

void
Cloud::SetProjectionPtr(typename CloudProjection::Ptr proj_ptr)
{
	_projection = proj_ptr;
}

void
Cloud::InitProjection(const ProjectionParams& params)
{
	if (_projection)
	{
		throw std::runtime_error("projection is already initialized");
	}
	_projection = CloudProjection::Ptr(new SphericalProjection(params));
	if (!_projection)
	{
		fprintf(stderr, "ERROR: failed to initalize projection.\n");
		return;
	}
	_projection = _projection->Clone();
	_projection->InitFromPoints(_points);
}

Cloud::Ptr
Cloud::FromImage(const cv::Mat& image, const ProjectionParams& params)
{
	CloudProjection::Ptr proj = CloudProjection::Ptr(new RingProjection(params));
	Cloud cloud;

	proj->CheckImageAndStorage(image);
	proj->CloneDepthImage(image);

	for (int r = 0; r < image.rows; ++r)
	{
		for (int c = 0; c < image.cols; ++c)
		{
			if (image.at<float>(r, c) < 0.0001f)
			{
				continue;
			}

			RichPoint point = proj->UnprojectPoint(image, r, c);

			point.setIntensity(-1);
			point.setElongation(-1);
			point.setScore(-1);

			cloud.push_back(point);
			proj->at(r, c).points().push_back(cloud.points().size() - 1);
		}
	}

	cloud.SetProjectionPtr(proj);

	return std::make_shared<Cloud>(cloud);
}

Cloud::Ptr
Cloud::FromImage(const cv::Mat& image_range, const cv::Mat& image_intensity,
		const cv::Mat& image_elongation, const std::shared_ptr<Score> score,
		const ProjectionParams& params)
{
	CloudProjection::Ptr proj = CloudProjection::Ptr(new RingProjection(params));
	Cloud cloud;

	proj->CheckImageAndStorage(image_range);
	proj->CloneDepthImage(image_range);
	proj->CloneElongationImage(image_elongation);

	for (int r = 0; r < image_range.rows; ++r)
	{
		for (int c = 0; c < image_range.cols; ++c)
		{
			if (image_range.at<float>(r, c) < 0.0001f)
			{
				continue;
			}

			RichPoint point = proj->UnprojectPoint(image_range, r, c);\
			float intensity_normalized = -1;
			float elongation_normalized = -1;

			if (image_intensity.rows >= image_range.rows
					&& image_intensity.cols >= image_range.cols)
			{
				intensity_normalized = image_intensity.at<float>(r, c)
						/ params.getProjectionParamsRaw()->intensity_norm_factor;
			}

			if (image_elongation.rows >= image_range.rows
					&& image_elongation.cols >= image_range.cols)
			{
				elongation_normalized = image_elongation.at<float>(r, c)
						/ params.getProjectionParamsRaw()->elongation_norm_factor;
			}

			point.setIntensity(intensity_normalized);
			point.setElongation(elongation_normalized);
			point.setScore(score->calculatePointScore(point));

			cloud.push_back(point);
			proj->at(r, c).points().push_back(cloud.points().size() - 1);
		}
	}

	cloud.SetProjectionPtr(proj);

	return std::make_shared<Cloud>(cloud);
}
}  // namespace depth_clustering
