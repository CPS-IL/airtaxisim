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

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <cassert>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "utils/cloud.h"
#include "utils/velodyne_utils.h"

namespace depth_clustering
{
using std::string;
using std::vector;

Cloud::Ptr
ReadKittiCloudTxt(const string& path)
{
	std::locale::global(std::locale("en_US.UTF-8"));
	fprintf(stderr, "Reading cloud from %s.\n", path.c_str());
	auto cloud_ptr = Cloud::Ptr(new Cloud);
	std::ifstream file(path.c_str());
	for (std::string line; std::getline(file, line, '\n');)
	{
		// here we parse the line
		vector<string> coords_str;
		boost::split(coords_str, line, boost::is_any_of(" "));
		if (coords_str.size() != 4)
		{
			fprintf(stderr, "ERROR: format of line is wrong.\n");
			continue;
		}
		RichPoint point;
		point.x() = std::stof(coords_str[0]);
		point.y() = std::stof(coords_str[1]);
		point.z() = std::stof(coords_str[2]);
		cloud_ptr->push_back(point);
	}
	return cloud_ptr;
}

Cloud::Ptr
CloudFromMat(const cv::Mat& image, const ProjectionParams& config)
{
	auto cloud_ptr = Cloud::Ptr(new Cloud);
	for (int row = 0; row < image.rows; ++row)
	{
		for (int col = 0; col < image.cols; ++col)
		{
			float meters = image.at<float>(row, col);
			float angle_xy = config.AngleFromCol(col).val();
			float angle_z = config.AngleFromRow(row).val();
			RichPoint point;
			point.x() = meters * cos(angle_z) * cos(angle_xy);
			point.y() = meters * cos(angle_z) * sin(angle_xy);
			point.z() = meters * sin(angle_z);
			point.ring() = row;
			cloud_ptr->push_back(point);
		}
	}
	return cloud_ptr;
}

Cloud::Ptr
ReadKittiCloud(const string& path)
{
	Cloud::Ptr cloud(new Cloud);
	std::fstream file(path.c_str(), std::ios::in | std::ios::binary);
	if (file.good())
	{
		file.seekg(0, std::ios::beg);
		float intensity = 0;
		for (int i = 0; file.good() && !file.eof(); ++i)
		{
			RichPoint point;
			file.read(reinterpret_cast<char*>(&point.x()), sizeof(float));
			file.read(reinterpret_cast<char*>(&point.y()), sizeof(float));
			file.read(reinterpret_cast<char*>(&point.z()), sizeof(float));
			// ignore intensity
			file.read(reinterpret_cast<char*>(&intensity), sizeof(float));
			cloud->push_back(point);
		}
		file.close();
	}
	return cloud;
}

cv::Mat
FixKITTIDepth(const cv::Mat& original)
{
	cv::Mat fixed = original;
	for (int r = 0; r < fixed.rows; ++r)
	{
		auto correction = MOOSMAN_CORRECTIONS[r];
		for (int c = 0; c < fixed.cols; ++c)
		{
			if (fixed.at<float>(r, c) < 0.001f)
			{
				continue;
			}
			fixed.at<float>(r, c) -= correction;
		}
	}
	return fixed;
}

cv::Mat
MatFromPNGCamera(const string& path)
{
	if (path == "")
	{
		return cv::Mat();
	}

#if CV_MAJOR_VERSION >= 3
	cv::Mat image_camera = cv::imread(path, cv::IMREAD_COLOR);
#else
	cv::Mat image_camera = cv::imread(path, CV_LOAD_IMAGE_COLOR);
#endif

	cv::cvtColor(image_camera, image_camera, cv::COLOR_BGR2RGB);

	return image_camera;
}

cv::Mat
MatFromPNGRange(const string& path, std::shared_ptr<ProjectionParams> projection_parameter)
{
	if (path == "")
	{
		return cv::Mat();
	}

#if CV_MAJOR_VERSION >= 3
	cv::Mat image_range = cv::imread(path, cv::IMREAD_ANYDEPTH);
#else
	cv::Mat image_range = cv::imread(path, CV_LOAD_IMAGE_ANYDEPTH);
#endif

	image_range.convertTo(image_range, CV_32F);
	image_range /= 500.0;

	return LimitHorizontalFieldOfView(FixKITTIDepth(image_range), projection_parameter);
}

cv::Mat
MatFromPNGIntensity(const string& path, std::shared_ptr<ProjectionParams> projection_parameter)
{
	if (path == "")
	{
		return cv::Mat();
	}

	std::cout << "[WARN]: The processing of \".png\" type intensity images is not implemented."
			<< std::endl;

	return cv::Mat();
}

cv::Mat
MatFromPNGElongation(const string& path, std::shared_ptr<ProjectionParams> projection_parameter)
{
	if (path == "")
	{
		return cv::Mat();
	}

	std::cout << "[WARN]: The processing of \".png\" type elongation images is not implemented."
			<< std::endl;

	return cv::Mat();
}

cv::Mat
MatFromTIFFRange(const string& path, std::shared_ptr<ProjectionParams> projection_parameter)
{
	if (path == "")
	{
		return cv::Mat();
	}

#if CV_MAJOR_VERSION >= 3
	cv::Mat image_range = cv::imread(path, cv::IMREAD_ANYDEPTH);
#else
	cv::Mat image_range = cv::imread(path, CV_LOAD_IMAGE_ANYDEPTH);
#endif

	image_range.convertTo(image_range, CV_32F);
	image_range /= 65535.0;
	image_range *= 100.0;

	return LimitHorizontalFieldOfView(image_range, projection_parameter);
}

cv::Mat
MatFromTIFFIntensity(const string& path, std::shared_ptr<ProjectionParams> projection_parameter)
{
	if (path == "")
	{
		return cv::Mat();
	}

#if CV_MAJOR_VERSION >= 3
	cv::Mat image_intensity = cv::imread(path, cv::IMREAD_ANYDEPTH);
#else
	cv::Mat image_intensity = cv::imread(path, CV_LOAD_IMAGE_ANYDEPTH);
#endif

	image_intensity.convertTo(image_intensity, CV_32F);
	image_intensity /= 65535.0;
	image_intensity *= projection_parameter->getProjectionParamsRaw()->intensity_norm_factor;

	return LimitHorizontalFieldOfView(image_intensity, projection_parameter);
}

cv::Mat
MatFromTIFFElongation(const string& path, std::shared_ptr<ProjectionParams> projection_parameter)
{
	if (path == "")
	{
		return cv::Mat();
	}

#if CV_MAJOR_VERSION >= 3
	cv::Mat image_elongation = cv::imread(path, cv::IMREAD_ANYDEPTH);
#else
	cv::Mat image_elongation = cv::imread(path, CV_LOAD_IMAGE_ANYDEPTH);
#endif

	image_elongation.convertTo(image_elongation, CV_32F);
	image_elongation /= 65535.0;
	image_elongation *= projection_parameter->getProjectionParamsRaw()->elongation_norm_factor;

	return LimitHorizontalFieldOfView(image_elongation, projection_parameter);
}

cv::Mat
LimitHorizontalFieldOfView(const cv::Mat& image,
		std::shared_ptr<ProjectionParams> projection_parameter)
{
	if (!projection_parameter)
	{
		return image;
	}

	return image(cv::Range(0, image.rows),
			cv::Range(projection_parameter->getProjectionParamsRaw()->horizontal_step_start,
					projection_parameter->getProjectionParamsRaw()->horizontal_step_end));
}
}  // namespace depth_clustering
