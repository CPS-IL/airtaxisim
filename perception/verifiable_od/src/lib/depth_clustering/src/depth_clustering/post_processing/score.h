/*
 * score.h
 *
 *  Created on: Oct 13, 2021
 *      Author: simonyu
 */

#ifndef DEPTH_CLUSTERING_POST_PROCESSING_SCORE_H_
#define DEPTH_CLUSTERING_POST_PROCESSING_SCORE_H_

#include "post_processing/bounding_box.h"
#include "utils/cloud.h"
#include "utils/rich_point.h"

namespace depth_clustering
{
class Score
{
public:

	enum class TypePoint
	{
		Type_1, Type_2
	};

	enum class TypeCluster
	{
		Type_1, Type_2
	};

	enum class TypeFrame
	{
		Type_1
	};

	Score();

	Score(const TypePoint& type_point, const TypeCluster& type_cluster,
			const TypeFrame& type_frame);

	void
	setTypePoint(const TypePoint& type_point);

	void
	setTypeCluster(const TypeCluster& type_cluster);

	void
	setTypeFrame(const TypeFrame& type_frame);

	float
	calculatePointScore(const RichPoint& point);

	float
	calculateClusterScore(const Cloud& cloud);

	float
	calculateFrameScore(const std::shared_ptr<BoundingBox::Frame<BoundingBox::Cluster>> frame);

private:

	float
	boundScore(const float& score);

	float
	calculatePointScoreType1(const RichPoint& point);

	float
	calculatePointScoreType2(const RichPoint& point);

	float
	calculateClusterScoreType1(const Cloud& cloud);

	float
	calculateClusterScoreType2(const Cloud& cloud);

	float
	calculateFrameScoreType1(const std::shared_ptr<BoundingBox::Frame<BoundingBox::Cluster>> frame);

	TypePoint type_point_;
	TypeCluster type_cluster_;
	TypeFrame type_frame_;
};
} // namespace depth_clustering

#endif /* DEPTH_CLUSTERING_POST_PROCESSING_SCORE_H_ */
