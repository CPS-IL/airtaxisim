/*
 * bounding_box.h
 *
 *  Created on: Sep 9, 2020
 *      Author: simonyu
 */

#ifndef SRC_POST_PROCESSING_BOUNDING_BOX_H_
#define SRC_POST_PROCESSING_BOUNDING_BOX_H_

#include <algorithm>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <vector>

#include "clusterers/abstract_clusterer.h"
#include "communication/abstract_client.h"
#include "ground_removal/depth_ground_remover.h"
#include "post_processing/camera_projection_parameter.h"
#include "utils/cloud.h"
#include "utils/timer.h"

using depth_clustering::Cloud;
using depth_clustering::time_utils::Timer;

namespace depth_clustering
{
class DepthClusteringParameter;
class CameraProjection;
class Score;

class BoundingBox: public depth_clustering::AbstractClient<std::unordered_map<uint16_t, Cloud>>
{
public:

	using AlignedEigenVectors =
	std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;
	template<typename Type>
	using Frame = std::vector<Type>;
	using Cluster = std::tuple<Cloud, float, std::string>; // <cloud, score, id>
	using Cube = std::tuple<Eigen::Vector3f, Eigen::Vector3f, float, float, std::string>; // <center, extent, rotation, score, id>
	using Polygon = std::tuple<AlignedEigenVectors, float, float, std::string>; // <hull, height, score, id>
	using Flat = std::tuple<Eigen::Vector2i, Eigen::Vector2i, float, float, std::string>; // <upper_left, lower_right, depth, score, id>

	enum class Type
	{
		Cube, Polygon, Flat
	};

	BoundingBox();

	BoundingBox(const std::shared_ptr<Score> score, const DepthClusteringParameter& parameter);

	BoundingBox(const std::shared_ptr<Score> score, const DepthClusteringParameter& parameter,
			const CameraProjectionParameter& camera_projection_parameter);

	std::shared_ptr<Frame<Cluster>>
	getFrameCluster() const;

	std::shared_ptr<Frame<Cube>>
	getFrameCube() const;

	std::shared_ptr<Frame<Polygon>>
	getFramePolygon() const;

	std::shared_ptr<Frame<Flat>>
	getFrameFlat() const;

	float
	getFrameScore() const;

	void
	setFrameCube(std::shared_ptr<Frame<Cube>> frame_cube);

	void
	setFramePolygon(std::shared_ptr<Frame<Polygon>> frame_polygon);

	void
	setFrameFlat(std::shared_ptr<Frame<Flat>> frame_flat);

	void
	clearFrames();

	void
	produceFrameFlat();

	void
	OnNewObjectReceived(const std::unordered_map<uint16_t, Cloud>& clouds, int id) override;

private:

	void
	CreateCubes(const Cluster& cluster);

	void
	CreatePolygons(const Cluster& cluster);

	Type type_;
	bool use_score_filter_;
	float score_filter_threshold_;
	float frame_score_;
	int id_;

	std::shared_ptr<Frame<Cluster>> frame_cluster_;
	std::shared_ptr<Frame<Cube>> frame_cube_;
	std::shared_ptr<Frame<Polygon>> frame_polygon_;
	std::shared_ptr<Frame<Flat>> frame_flat_;
	std::shared_ptr<CameraProjection> camera_projection_;
	std::shared_ptr<Score> score_;
};
} // namespace depth_clustering

#endif /* SRC_POST_PROCESSING_BOUNDING_BOX_H_ */
