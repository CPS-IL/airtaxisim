/*
 * depth_clustering_parameter.h
 *
 *  Created on: Sep 11, 2020
 *      Author: simonyu
 */

#ifndef SRC_API_PARAMETER_H_
#define SRC_API_PARAMETER_H_

#include "image_labelers/diff_helpers/diff_factory.h"
#include "post_processing/bounding_box.h"
#include "post_processing/score.h"
#include "utils/radians.h"

namespace depth_clustering
{
struct DepthClusteringParameter
{
	float distance_clustering;
	float score_clustering;
	Radians angle_clustering;
	Radians angle_ground_removal;
	int size_cluster_min;
	int size_cluster_max;
	int size_smooth_window;
	bool use_camera_fov;
	bool use_score_filter;
	float score_filter_threshold;
	Score::TypePoint score_type_point;
	Score::TypeCluster score_type_cluster;
	Score::TypeFrame score_type_frame;
	BoundingBox::Type bounding_box_type;
	DiffFactory::DiffType difference_type;
	std::string dataset_file_type;
	std::string dataset_name;
	std::string ground_truth_cube_file_name;
	std::string ground_truth_flat_file_name;

	DepthClusteringParameter() :
			distance_clustering(0.17), score_clustering(0.2), angle_clustering(10_deg), angle_ground_removal(
					9_deg), size_cluster_min(10), size_cluster_max(20000), size_smooth_window(5), use_camera_fov(
					false), use_score_filter(false), score_filter_threshold(0), score_type_point(
					Score::TypePoint::Type_1), score_type_cluster(Score::TypeCluster::Type_1), score_type_frame(
					Score::TypeFrame::Type_1), bounding_box_type(BoundingBox::Type::Cube), difference_type(
					DiffFactory::DiffType::ANGLES_PRECOMPUTED), dataset_file_type(".tiff"), dataset_name(
					""), ground_truth_cube_file_name("ground_truth_cube.json"), ground_truth_flat_file_name(
					"ground_truth_flat.json")
	{
	}
};
} // namespace depth_clustering

#endif /* SRC_API_PARAMETER_H_ */
