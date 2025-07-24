/*
 * score.cpp
 *
 *  Created on: Oct 13, 2021
 *      Author: simonyu
 */

#include <algorithm>

#include "post_processing/score.h"

namespace depth_clustering
{
Score::Score() :
		type_point_(TypePoint::Type_1), type_cluster_(TypeCluster::Type_1), type_frame_(
				TypeFrame::Type_1)
{
}

Score::Score(const TypePoint& type_point, const TypeCluster& type_cluster,
		const TypeFrame& type_frame) :
		type_point_(type_point), type_cluster_(type_cluster), type_frame_(type_frame)
{
}

void
Score::setTypePoint(const TypePoint& type_point)
{
	type_point_ = type_point;
}

void
Score::setTypeCluster(const TypeCluster& type_cluster)
{
	type_cluster_ = type_cluster;
}

void
Score::setTypeFrame(const TypeFrame& type_frame)
{
	type_frame_ = type_frame;
}

float
Score::calculatePointScore(const RichPoint& point)
{
	switch (type_point_)
	{
	case TypePoint::Type_1:
	{
		return calculatePointScoreType1(point);
	}
	case TypePoint::Type_2:
	{
		return calculatePointScoreType2(point);
	}
	default:
	{
		std::cout << "[WARN]: Unknown point score type." << std::endl;
		return calculatePointScoreType1(point);
	}
	}
}

float
Score::calculateClusterScore(const Cloud& cloud)
{
	switch (type_cluster_)
	{
	case TypeCluster::Type_1:
	{
		return calculateClusterScoreType1(cloud);
	}
	case TypeCluster::Type_2:
	{
		return calculateClusterScoreType2(cloud);
	}
	default:
	{
		std::cout << "[WARN]: Unknown cluster score type." << std::endl;
		return calculateClusterScoreType1(cloud);
	}
	}
}

float
Score::calculateFrameScore(const std::shared_ptr<BoundingBox::Frame<BoundingBox::Cluster>> frame)
{
	switch (type_frame_)
	{
	case TypeFrame::Type_1:
	{
		return calculateFrameScoreType1(frame);
	}
	default:
	{
		std::cout << "[WARN]: Unknown frame score type." << std::endl;
		return calculateFrameScoreType1(frame);
	}
	}
}

float
Score::boundScore(const float& score)
{
	return std::min<float>(1, std::max<float>(0, score));
}

float
Score::calculatePointScoreType1(const RichPoint& point)
{
	/*
	 * 1 - elongation
	 */

	float point_score = -1;

	if (point.elongation() < 0)
	{
		return point_score;
	}

	point_score = 1 - point.elongation();
	point_score = boundScore(point_score);

	return point_score;

}

float
Score::calculatePointScoreType2(const RichPoint& point)
{
	/*
	 * min(1, intensity + (1 - elongation))
	 */

	float point_score = -1;

	if (point.intensity() < 0 || point.elongation() < 0)
	{
		return point_score;
	}

	point_score = point.intensity() + (1 - point.elongation());
	point_score = boundScore(point_score);

	return point_score;
}

float
Score::calculateClusterScoreType1(const Cloud& cloud)
{
	/*
	 * Weighted average on point scores over depth
	 */

	int point_counter_invalid = 0;
	float weight_min = std::numeric_limits<float>::max();
	float weight_max = std::numeric_limits<float>::min();
	float point_score_total_weighted = 0;
	float weight_total = 0;
	float cluster_score = -1;

	for (const auto &point : cloud.points())
	{
		if (point.AsEigenVector().norm() < weight_min)
		{
			weight_min = point.AsEigenVector().norm();
		}

		if (point.AsEigenVector().norm() > weight_max)
		{
			weight_max = point.AsEigenVector().norm();
		}
	}

	for (const auto &point : cloud.points())
	{
		if (point.score() < 0 || point.score() > 1)
		{
			point_counter_invalid++;
			continue;
		}

		float weight = 1 - (point.AsEigenVector().norm() - weight_min) / (weight_max - weight_min);

		point_score_total_weighted += weight * point.score();
		weight_total += weight;
	}

	if (point_counter_invalid < static_cast<int>(cloud.points().size()))
	{
		cluster_score = point_score_total_weighted / weight_total;
		cluster_score = boundScore(cluster_score);
	}

	return cluster_score;
}

float
Score::calculateClusterScoreType2(const Cloud& cloud)
{
	/*
	 * Average on point scores
	 */

	int point_counter_invalid = 0;
	float point_score_total = 0;
	float cluster_score = -1;

	for (const auto &point : cloud.points())
	{
		if (point.score() < 0 || point.score() > 1)
		{
			point_counter_invalid++;
			continue;
		}

		point_score_total += point.score();
	}

	if (point_counter_invalid < static_cast<int>(cloud.points().size()))
	{
		cluster_score = point_score_total / static_cast<int>(cloud.points().size());
		cluster_score = boundScore(cluster_score);
	}

	return cluster_score;
}

float
Score::calculateFrameScoreType1(
		const std::shared_ptr<BoundingBox::Frame<BoundingBox::Cluster>> frame)
{
	/*
	 * Cluster score average
	 */

	int cluster_counter_invalid = 0;
	float cluster_score_total = 0;
	float frame_score = -1;

	for (const auto &cluster : *frame)
	{
		if (std::get<1>(cluster) < 0 || std::get<1>(cluster) > 1)
		{
			cluster_counter_invalid++;
			continue;
		}

		cluster_score_total += std::get<1>(cluster);
	}

	if (cluster_counter_invalid < static_cast<int>(frame->size()))
	{
		frame_score = cluster_score_total / static_cast<int>(frame->size());
		frame_score = boundScore(frame_score);
	}

	return frame_score;
}
} // namespace depth_clustering
