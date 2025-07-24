/*
 * logger.cpp
 *
 *  Created on: Sep 9, 2020
 *      Author: simonyu
 */

#include <boost/property_tree/json_parser.hpp>

#include "post_processing/logger.h"

namespace depth_clustering
{

Logger::Logger() :
		Logger(LoggerParameter())
{
}

Logger::Logger(const LoggerParameter& parameter) :
		parameter_(parameter)
{
}

void
Logger::setBoundingBox(std::shared_ptr<BoundingBox> bounding_box)
{
	bounding_box_ = bounding_box;
}

void
Logger::logBoundingBoxFrame(const std::string& frame_name,
		const BoundingBox::Type& bounding_box_type)
{
	switch (bounding_box_type)
	{
	case BoundingBox::Type::Cube:
	{
		logBoundingBoxFrameCube(frame_name);
		break;
	}
	case BoundingBox::Type::Polygon:
	{
		logBoundingBoxFramePolygon(frame_name);
		break;
	}
	case BoundingBox::Type::Flat:
	{
		logBoundingBoxFrameFlat(frame_name);
		break;
	}
	default:
	{
		std::cout << "[WARN]: Unknown bounding box type." << std::endl;
		logBoundingBoxFrameCube(frame_name);
		break;
	}
	}
}

void
Logger::logBoundingBoxFrameCube(const std::string& frame_name)
{
	if (!parameter_.log)
	{
		return;
	}

	if (!bounding_box_)
	{
		std::cout << "[ERROR]: Bounding box missing." << std::endl;
		return;
	}

	boost::property_tree::ptree bounding_box_cube_frame_array_value;
	auto bounding_box_frame_cube = bounding_box_->getFrameCube();
	const auto bounding_box_frame_cube_score = bounding_box_->getFrameScore();

	if (!bounding_box_frame_cube)
	{
		std::cout << "[ERROR]: Cube bounding box frame missing." << std::endl;
		return;
	}

	auto bounding_box_cube_frame_array_optional = bounding_box_log_tree_cube_.get_child_optional(
			boost::property_tree::ptree::path_type(frame_name, '/'));

	if (!bounding_box_cube_frame_array_optional)
	{
		bounding_box_log_tree_cube_.add_child(
				boost::property_tree::ptree::path_type(frame_name, '/'),
				boost::property_tree::ptree());
	}

	for (const auto &bounding_box_cube : *bounding_box_frame_cube)
	{
		boost::property_tree::ptree bounding_box_cube_array_value;
		boost::property_tree::ptree bounding_box_cube_array;

		auto center = std::get<0>(bounding_box_cube);
		auto extent = std::get<1>(bounding_box_cube);
		auto rotation = std::get<2>(bounding_box_cube);
		auto score = std::get<3>(bounding_box_cube);
		auto id = std::get<4>(bounding_box_cube);
		auto &bounding_box_cube_frame_array = bounding_box_log_tree_cube_.get_child(
				boost::property_tree::ptree::path_type(frame_name, '/'));

		bounding_box_cube_array_value.put_value(center.x());
		bounding_box_cube_array.push_back(std::make_pair("", bounding_box_cube_array_value));

		bounding_box_cube_array_value.put_value(center.y());
		bounding_box_cube_array.push_back(std::make_pair("", bounding_box_cube_array_value));

		bounding_box_cube_array_value.put_value(center.z());
		bounding_box_cube_array.push_back(std::make_pair("", bounding_box_cube_array_value));

		bounding_box_cube_array_value.put_value(extent.x());
		bounding_box_cube_array.push_back(std::make_pair("", bounding_box_cube_array_value));

		bounding_box_cube_array_value.put_value(extent.y());
		bounding_box_cube_array.push_back(std::make_pair("", bounding_box_cube_array_value));

		bounding_box_cube_array_value.put_value(extent.z());
		bounding_box_cube_array.push_back(std::make_pair("", bounding_box_cube_array_value));

		bounding_box_cube_array_value.put_value(rotation);
		bounding_box_cube_array.push_back(std::make_pair("", bounding_box_cube_array_value));

		bounding_box_cube_array_value.put_value(score);
		bounding_box_cube_array.push_back(std::make_pair("", bounding_box_cube_array_value));

		bounding_box_cube_array_value.put_value(id);
		bounding_box_cube_array.push_back(std::make_pair("", bounding_box_cube_array_value));

		bounding_box_cube_frame_array.push_back(std::make_pair("", bounding_box_cube_array));
	}

	auto &bounding_box_cube_frame_array = bounding_box_log_tree_cube_.get_child(
			boost::property_tree::ptree::path_type(frame_name, '/'));

	bounding_box_cube_frame_array_value.put_value(bounding_box_frame_cube_score);
	bounding_box_cube_frame_array.push_back(
			std::make_pair("", bounding_box_cube_frame_array_value));
}

void
Logger::logBoundingBoxFramePolygon(const std::string& frame_name)
{
	if (!parameter_.log)
	{
		return;
	}

	if (!bounding_box_)
	{
		std::cout << "[ERROR]: Bounding box missing." << std::endl;
		return;
	}

	boost::property_tree::ptree bounding_box_polygon_frame_array_value;
	auto bounding_box_frame_polygon = bounding_box_->getFramePolygon();
	const auto bounding_box_frame_polygon_score = bounding_box_->getFrameScore();

	if (!bounding_box_frame_polygon)
	{
		std::cout << "[ERROR]: Polygon bounding box frame missing." << std::endl;
		return;
	}

	auto bounding_box_polygon_frame_array_optional =
			bounding_box_log_tree_polygon_.get_child_optional(
					boost::property_tree::ptree::path_type(frame_name, '/'));

	if (!bounding_box_polygon_frame_array_optional)
	{
		bounding_box_log_tree_polygon_.add_child(
				boost::property_tree::ptree::path_type(frame_name, '/'),
				boost::property_tree::ptree());
	}

	for (const auto &bounding_box_polygon : *bounding_box_frame_polygon)
	{
		boost::property_tree::ptree bounding_box_polygon_array_value;
		boost::property_tree::ptree bounding_box_polygon_array;
		boost::property_tree::ptree bounding_box_polygon_hull_vector_array_value;
		boost::property_tree::ptree bounding_box_polygon_hull_vector_array;
		boost::property_tree::ptree bounding_box_polygon_hull_array;

		auto hull = std::get<0>(bounding_box_polygon);
		auto height = std::get<1>(bounding_box_polygon);
		auto score = std::get<2>(bounding_box_polygon);
		auto id = std::get<3>(bounding_box_polygon);
		auto &bounding_box_polygon_frame_array = bounding_box_log_tree_polygon_.get_child(
				boost::property_tree::ptree::path_type(frame_name, '/'));

		for (const auto &hull_vector : hull)
		{
			bounding_box_polygon_hull_vector_array_value.put_value(hull_vector.x());
			bounding_box_polygon_hull_vector_array.push_back(
					std::make_pair("", bounding_box_polygon_hull_vector_array_value));

			bounding_box_polygon_hull_vector_array_value.put_value(hull_vector.y());
			bounding_box_polygon_hull_vector_array.push_back(
					std::make_pair("", bounding_box_polygon_hull_vector_array_value));

			bounding_box_polygon_hull_vector_array_value.put_value(hull_vector.z());
			bounding_box_polygon_hull_vector_array.push_back(
					std::make_pair("", bounding_box_polygon_hull_vector_array_value));

			bounding_box_polygon_hull_array.push_back(
					std::make_pair("", bounding_box_polygon_hull_vector_array));

			bounding_box_polygon_hull_vector_array.clear();
		}

		bounding_box_polygon_array.push_back(std::make_pair("", bounding_box_polygon_hull_array));

		bounding_box_polygon_array_value.put_value(height);
		bounding_box_polygon_array.push_back(std::make_pair("", bounding_box_polygon_array_value));

		bounding_box_polygon_array_value.put_value(score);
		bounding_box_polygon_array.push_back(std::make_pair("", bounding_box_polygon_array_value));

		bounding_box_polygon_array_value.put_value(id);
		bounding_box_polygon_array.push_back(std::make_pair("", bounding_box_polygon_array_value));

		bounding_box_polygon_frame_array.push_back(std::make_pair("", bounding_box_polygon_array));
	}

	auto &bounding_box_polygon_frame_array = bounding_box_log_tree_polygon_.get_child(
			boost::property_tree::ptree::path_type(frame_name, '/'));

	bounding_box_polygon_frame_array_value.put_value(bounding_box_frame_polygon_score);
	bounding_box_polygon_frame_array.push_back(
			std::make_pair("", bounding_box_polygon_frame_array_value));
}

void
Logger::logBoundingBoxFrameFlat(const std::string& frame_name)
{
	if (!parameter_.log)
	{
		return;
	}

	if (!bounding_box_)
	{
		std::cout << "[ERROR]: Bounding box missing." << std::endl;
		return;
	}

	boost::property_tree::ptree bounding_box_flat_frame_array_value;
	auto bounding_box_frame_flat = bounding_box_->getFrameFlat();
	const auto bounding_box_frame_flat_score = bounding_box_->getFrameScore();

	if (!bounding_box_frame_flat)
	{
		std::cout << "[ERROR]: Flat bounding box frame missing." << std::endl;
		return;
	}

	auto bounding_box_flat_frame_array_optional = bounding_box_log_tree_flat_.get_child_optional(
			boost::property_tree::ptree::path_type(frame_name, '/'));

	if (!bounding_box_flat_frame_array_optional)
	{
		bounding_box_log_tree_flat_.add_child(
				boost::property_tree::ptree::path_type(frame_name, '/'),
				boost::property_tree::ptree());
	}

	for (const auto &bounding_box_flat : *bounding_box_frame_flat)
	{
		boost::property_tree::ptree bounding_box_flat_array_value;
		boost::property_tree::ptree bounding_box_flat_array;

		auto corner_upper_left = std::get<0>(bounding_box_flat);
		auto corner_lower_right = std::get<1>(bounding_box_flat);
		auto depth = std::get<2>(bounding_box_flat);
		auto score = std::get<3>(bounding_box_flat);
		auto id = std::get<4>(bounding_box_flat);
		auto &bounding_box_flat_frame_array = bounding_box_log_tree_flat_.get_child(
				boost::property_tree::ptree::path_type(frame_name, '/'));

		bounding_box_flat_array_value.put_value(corner_upper_left.x());
		bounding_box_flat_array.push_back(std::make_pair("", bounding_box_flat_array_value));

		bounding_box_flat_array_value.put_value(corner_upper_left.y());
		bounding_box_flat_array.push_back(std::make_pair("", bounding_box_flat_array_value));

		bounding_box_flat_array_value.put_value(corner_lower_right.x());
		bounding_box_flat_array.push_back(std::make_pair("", bounding_box_flat_array_value));

		bounding_box_flat_array_value.put_value(corner_lower_right.y());
		bounding_box_flat_array.push_back(std::make_pair("", bounding_box_flat_array_value));

		bounding_box_flat_array_value.put_value(depth);
		bounding_box_flat_array.push_back(std::make_pair("", bounding_box_flat_array_value));

		bounding_box_flat_array_value.put_value(score);
		bounding_box_flat_array.push_back(std::make_pair("", bounding_box_flat_array_value));

		bounding_box_flat_array_value.put_value(id);
		bounding_box_flat_array.push_back(std::make_pair("", bounding_box_flat_array_value));

		bounding_box_flat_frame_array.push_back(std::make_pair("", bounding_box_flat_array));
	}

	auto &bounding_box_flat_frame_array = bounding_box_log_tree_flat_.get_child(
			boost::property_tree::ptree::path_type(frame_name, '/'));

	bounding_box_flat_frame_array_value.put_value(bounding_box_frame_flat_score);
	bounding_box_flat_frame_array.push_back(
			std::make_pair("", bounding_box_flat_frame_array_value));
}

void
Logger::writeBoundingBoxLog(const BoundingBox::Type& bounding_box_type)
{
	if (!parameter_.log)
	{
		return;
	}

	if (parameter_.log_path != "" && parameter_.log_path[parameter_.log_path.size() - 1] != '/')
	{
		parameter_.log_path += "/";
	}

	std::ofstream bounding_box_log_file;

	bounding_box_log_file.close();

	switch (bounding_box_type)
	{
	case BoundingBox::Type::Cube:
	{
		bounding_box_log_file.open(parameter_.log_path + parameter_.log_file_name_cube,
				std::fstream::out | std::fstream::trunc);

		boost::property_tree::write_json(bounding_box_log_file, bounding_box_log_tree_cube_);

		std::cout << "[INFO]: Wrote to log file \""
				<< parameter_.log_path + parameter_.log_file_name_cube << "\"." << std::endl;

		break;
	}
	case BoundingBox::Type::Polygon:
	{
		bounding_box_log_file.open(parameter_.log_path + parameter_.log_file_name_polygon,
				std::fstream::out | std::fstream::trunc);

		boost::property_tree::write_json(bounding_box_log_file, bounding_box_log_tree_polygon_);

		std::cout << "[INFO]: Wrote to log file \""
				<< parameter_.log_path + parameter_.log_file_name_polygon << "\"." << std::endl;

		break;
	}
	case BoundingBox::Type::Flat:
	{
		bounding_box_log_file.open(parameter_.log_path + parameter_.log_file_name_flat,
				std::fstream::out | std::fstream::trunc);

		boost::property_tree::write_json(bounding_box_log_file, bounding_box_log_tree_flat_);

		std::cout << "[INFO]: Wrote to log file \""
				<< parameter_.log_path + parameter_.log_file_name_flat << "\"." << std::endl;

		break;
	}
	default:
	{
		std::cout << "[WARN]: Unknown bounding box type." << std::endl;

		bounding_box_log_file.open(parameter_.log_path + parameter_.log_file_name_cube,
				std::fstream::out | std::fstream::trunc);

		boost::property_tree::write_json(bounding_box_log_file, bounding_box_log_tree_cube_);

		std::cout << "[INFO]: Wrote to log file \""
				<< parameter_.log_path + parameter_.log_file_name_cube << "\"." << std::endl;

		break;
	}
	}

	bounding_box_log_file.close();
}

} // namespace depth_clustering
