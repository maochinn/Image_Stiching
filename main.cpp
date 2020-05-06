#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/features2d/features2d.hpp>

#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <algorithm>

#include "Features.h"
#include "FeatureDetector.h"

using namespace cv;
using namespace std;


Mat imageStitching(vector<Mat> images, vector<float> focal_lengths,
	vector<vector<KeyPoint>> images_keypoints, vector<vector<DMatch>> keypoints_matchs);

Point2f cylindricalProjection(Point2f p, int wdt, int hgt, float f)
{
	float s = f;

	float center_x = wdt * 0.5f;
	float center_y = hgt * 0.5f;

	float x = p.x - center_x;
	float y = p.y - center_y;
	float theta = atan(x / f);
	float h = y / (sqrt(x * x + f * f));

	return Point2f(s * theta + center_x, s * h + center_y);
}

//Mat cylindricalProjection(Mat image, float f)
//{
//	float s = f;
//	Mat projected(image.rows, image.cols, CV_8UC3);
//	float center_x = image.cols * 0.5f;
//	float center_y = image.rows * 0.5f;
//	for (int i = 0; i < image.cols; i++)
//		for (int j = 0;j < image.rows; j++)
//		{
//			float x = i - center_x;
//			float y = j - center_y;
//			float theta = atan(x / f);
//			float h = y / (sqrt(x * x + f * f));
//
//			projected.at<Vec3b>(s * h + center_y, s * theta + center_x) = image.at<Vec3b>(j, i);
//		}
//	return projected;
//}
void loadImageList(String path, vector<Mat>& images, vector<float>& focal_lengths);

int main(int argc, char** argv)
{
	vector<Mat> images;
	vector<float> focal_lengths;
	string path;

	std::cout << "scan your images folder(include images and list.txt):";
	std::cin >> path;
	loadImageList(path, images, focal_lengths);
	if (images.empty())
	{
		puts("load file error!");
		return 1;
	}
	images.push_back(images[0]);
	focal_lengths.push_back(focal_lengths[0]);

	for (auto& img : images)
	{
		pyrDown(img, img);
		pyrDown(img, img);
		pyrDown(img, img);
	}

	//Ptr<FastFeatureDetector> fast = FastFeatureDetector::create(60);
	//Ptr<DescriptorExtractor> extractor = BRISK::create();
	//Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	
	vector<vector<KeyPoint>> imgs_keypoints;
	//vector<Mat> imgs_descriptions;
	vector<vector<DMatch>> imgs_matchs;

	std::vector<std::vector<FeaturePoint>> img_features;
	for (Mat& image : images)
	{
		std::vector<FeaturePoint> features;
		DetectFeature(image, features, 5, 2, 60.0f);
		img_features.push_back(features);
	}
	//search best matching for other images
	for (std::vector<FeaturePoint>& features : img_features)
		for (FeaturePoint& feature : features)
			for (int i = 0; i < images.size(); i++)
			{
				feature.bestMatch.push_back(-1);
				feature.bestMatchDistance.push_back(-1);
			}
	for (int i = 0; i < img_features.size(); i++)
		for (int j = 0; j < img_features.size(); j++)
			for (int p0 = 0; p0 < img_features[i].size(); p0++)
			{
				float min = FLT_MAX;
				float sec_min = FLT_MAX;
				for (int p1 = 0; p1 < img_features[j].size(); p1++)
				{
					if (img_features[i][p0].level != img_features[j][p1].level)
						continue;
					float distance = 0.0f;
					for (int k = 0; k < 64; k++)
					{
						float diff = img_features[i][p0].descriptor[k] - img_features[j][p1].descriptor[k];
						distance += diff * diff;
					}
					if (distance < min)
					{
						sec_min = min;
						min = distance;
						img_features[i][p0].bestMatch[j] = p1;
						img_features[i][p0].bestMatchDistance[j] = min;
					}
				}
				if (min / sec_min > 0.5f)
				{
					//discard this match
					img_features[i][p0].bestMatch[j] = -1;
					img_features[i][p0].bestMatchDistance[j] = -1;
				}
			}
	//convert to opencv data structure
	for (std::vector<FeaturePoint>& features : img_features)
	{
		vector<KeyPoint> keypoints;
		for (FeaturePoint& feature : features)
		{
			keypoints.push_back(KeyPoint(feature.x, feature.y, feature.level, feature.orientation, feature.response));
		}
		imgs_keypoints.push_back(keypoints);
	}
	for (int i = 0; i < images.size() - 1; i++)
	{
		vector<DMatch> matches;
		for (int queryIdx = 0; queryIdx < img_features[i].size(); queryIdx++)
		{
			DMatch match(queryIdx, img_features[i][queryIdx].bestMatch[i+1], img_features[i][queryIdx].bestMatchDistance[i+1]);
			if (match.trainIdx < 0)
				continue;
			matches.push_back(match);
		}

		vector<DMatch> samples;
		vector<float> distances;
		for (DMatch match : matches)
		{
			distances.push_back(match.distance);
		}
		sort(distances.begin(), distances.end());
		int threshold = distances.size() * 0.5f;
		for (DMatch match : matches)
		{
			//if (match.distance < distances[threshold])
				samples.push_back(match);
		}
		imgs_matchs.push_back(samples);
	}

	Mat global_image = imageStitching(images, focal_lengths, imgs_keypoints, imgs_matchs);
	cv::imwrite("Stiching.png", global_image);
	imshow("Image Stiching", global_image);
	waitKey(0);
	return 0;
}
void loadImageList(String path, vector<Mat>& images, vector<float>& focal_lengths)
{
	path = path + "/";
	ifstream list_file((path + "list.txt").c_str());
	string name;
	float val;
	while (list_file >> name >> val) {
		Mat img = imread(path + name);
		images.push_back(img);
		focal_lengths.push_back(val);
	}
	list_file.close();
}

Mat findTranslation(
	vector<KeyPoint>& points_1, vector<KeyPoint>& points_2, vector<DMatch> matchs)
{
	float x_diff = 0.0f;
	float y_diff = 0.0f;
	for (DMatch& match : matchs)
	{
		float xi = points_1[match.queryIdx].pt.x;
		float yi = points_1[match.queryIdx].pt.y;

		float xj = points_2[match.trainIdx].pt.x;
		float yj = points_2[match.trainIdx].pt.y;

		x_diff += xi - xj;
		y_diff += yi - yj;
	}
	int n = matchs.size();
	Mat transform(cv::Size(3, 3), CV_64FC1);
	transform.at<double>(0, 0) = 1.0;
	transform.at<double>(0, 1) = 0.0;
	transform.at<double>(0, 2) = x_diff / n;
	transform.at<double>(1, 0) = 0.0;
	transform.at<double>(1, 1) = 1.0;
	transform.at<double>(1, 2) = y_diff / n;
	transform.at<double>(2, 0) = 0.0;
	transform.at<double>(2, 1) = 0.0;
	transform.at<double>(2, 2) = 1.0;

	return transform;
}
Mat findAffineTransform(vector<KeyPoint>& points_1, vector<KeyPoint>& points_2, vector<DMatch> matchs)
{
	Mat A = Mat::zeros(cv::Size(6, matchs.size() * 2), CV_64FC1);
	Mat b = Mat::zeros(cv::Size(1, matchs.size() * 2), CV_64FC1);

	for (int i = 0; i < matchs.size(); i++)
	{
		A.at<double>(2 * i, 0) = points_2[matchs[i].trainIdx].pt.x;
		A.at<double>(2 * i, 1) = points_2[matchs[i].trainIdx].pt.y;
		A.at<double>(2 * i, 2) = 1.0f;

		A.at<double>(2 * i + 1, 3) = points_2[matchs[i].trainIdx].pt.x;
		A.at<double>(2 * i + 1, 4) = points_2[matchs[i].trainIdx].pt.y;
		A.at<double>(2 * i + 1, 5) = 1.0f;
	}

	for (int i = 0; i < matchs.size(); i++)
	{
		b.at<double>(2 * i, 0) = points_1[matchs[i].queryIdx].pt.x;
		b.at<double>(2 * i + 1, 0) = points_1[matchs[i].queryIdx].pt.y;
	}

	Mat AT_A = A.t() * A;
	Mat AT_b = A.t() * b;
	Mat AT_A_inversed;
	invert(AT_A, AT_A_inversed, DECOMP_SVD);
	Mat x = AT_A_inversed * AT_b;

	Mat transform(cv::Size(3, 3), CV_64FC1);
	transform.at<double>(0, 0) = x.at<double>(0, 0);
	transform.at<double>(0, 1) = x.at<double>(1, 0);
	transform.at<double>(0, 2) = x.at<double>(2, 0);
	transform.at<double>(1, 0) = x.at<double>(3, 0);
	transform.at<double>(1, 1) = x.at<double>(4, 0);
	transform.at<double>(1, 2) = x.at<double>(5, 0);
	transform.at<double>(2, 0) = 0.0;
	transform.at<double>(2, 1) = 0.0;
	transform.at<double>(2, 2) = 1.0;

	return transform;
}
Mat findProjectiveTransform(vector<KeyPoint>& points_1, vector<KeyPoint>& points_2, vector<DMatch> matchs)
{
	Mat A = Mat::zeros(cv::Size(8, matchs.size() * 2), CV_64FC1);
	Mat b = Mat::zeros(cv::Size(1, matchs.size() * 2), CV_64FC1);

	for (int i = 0; i < matchs.size(); i++)
	{
		A.at<double>(2 * i, 0) = points_2[matchs[i].trainIdx].pt.x;
		A.at<double>(2 * i, 1) = points_2[matchs[i].trainIdx].pt.y;
		A.at<double>(2 * i, 2) = 1.0f;

		A.at<double>(2 * i + 1, 3) = points_2[matchs[i].trainIdx].pt.x;
		A.at<double>(2 * i + 1, 4) = points_2[matchs[i].trainIdx].pt.y;
		A.at<double>(2 * i + 1, 5) = 1.0f;
	}

	for (int i = 0; i < matchs.size(); i++)
	{
		b.at<double>(2 * i, 0) = points_1[matchs[i].queryIdx].pt.x;
		b.at<double>(2 * i + 1, 0) = points_1[matchs[i].queryIdx].pt.y;
	}

	Mat AT_A = A.t() * A;
	Mat AT_b = A.t() * b;
	Mat AT_A_inversed;
	invert(AT_A, AT_A_inversed, DECOMP_SVD);
	Mat x = AT_A_inversed * AT_b;

	Mat transform(cv::Size(3, 3), CV_64FC1);
	transform.at<double>(0, 0) = x.at<double>(0, 0);
	transform.at<double>(0, 1) = x.at<double>(1, 0);
	transform.at<double>(0, 2) = x.at<double>(2, 0);
	transform.at<double>(1, 0) = x.at<double>(3, 0);
	transform.at<double>(1, 1) = x.at<double>(4, 0);
	transform.at<double>(1, 2) = x.at<double>(5, 0);
	transform.at<double>(2, 0) = 0.0;
	transform.at<double>(2, 1) = 0.0;
	transform.at<double>(2, 2) = 1.0;

	return transform;
}

Mat imageStitching(vector<Mat> images, vector<float> focal_lengths, 
	vector<vector<KeyPoint>> images_keypoints, vector<vector<DMatch>> keypoints_matchs)
{
	// Warp to cylindrical coordinate
	vector<Mat> cylindrical_images;
	for (int i = 0;i<images.size();i++)
	{
		Mat projected(images[i].rows, images[i].cols, CV_8UC4);
		for (int x = 0; x < projected.cols; x++)
			for (int y = 0; y < projected.rows; y++)
			{
				projected.at<Vec4b>(y, x) = Vec4b(0, 0, 0, 0);
			}
		float center_x = images[i].cols * 0.5f;
		float center_y = images[i].rows * 0.5f;
		for (int x = 0; x < images[i].cols; x++)
			for (int y = 0; y < images[i].rows; y++)
			{
				Vec3f temp = images[i].at<Vec3b>(y, x);
				projected.at<Vec4b>(
					cylindricalProjection(Point2f(x, y), images[i].cols, images[i].rows, focal_lengths[i])) = 
					Vec4b(temp[0], temp[1], temp[2], 255);
			}
		cylindrical_images.push_back(projected);
		for (KeyPoint& p : images_keypoints[i])
		{
			p.pt = cylindricalProjection(p.pt, images[i].cols, images[i].rows, focal_lengths[i]);
		}
		
	}
	vector<Mat> transforms;
	//transform of first image is identity
	transforms.push_back(Mat::eye(3, 3, CV_64FC1));

	float P = 0.99;
	float p = 0.3;
	int n = 3;
	float k = log(1 - P) / log(1 - pow(p, n));
	float c = 100.0f;

	for (int i = 0; i < keypoints_matchs.size(); i++)
	{
		Mat max_transform;
		int max_inlier_count = 0;
		vector<DMatch> max_matchs;

		for (int j = 0; j < k; j++)
		{
			vector<DMatch> sample_matchs;
			vector<int> randoms;
			for (int temp = 0; temp < n; temp++)
			{
				int random = rand() % keypoints_matchs[i].size();
				if (randoms.size() >= keypoints_matchs[i].size())
					break;
				if (find(randoms.begin(), randoms.end(), random) != randoms.end())
				{
					temp--;
					continue;
				}

				randoms.push_back(random);
				sample_matchs.push_back(keypoints_matchs[i][random]);
			}
			//Mat transform = findAffineTransform(images_keypoints[i], images_keypoints[i + 1], sample_matchs);
			Mat transform = findTranslation(images_keypoints[i], images_keypoints[i + 1], sample_matchs);

			int inlier_count = 0;
			for (DMatch match : keypoints_matchs[i])
			{
				Mat p0(cv::Size(1, 3), CV_64FC1);
				Mat p1(cv::Size(1, 3), CV_64FC1);

				p0.at<double>(0, 0) = images_keypoints[i][match.queryIdx].pt.x;
				p0.at<double>(1, 0) = images_keypoints[i][match.queryIdx].pt.y;
				p0.at<double>(2, 0) = 1.0;
				p1.at<double>(0, 0) = images_keypoints[i + 1][match.trainIdx].pt.x;
				p1.at<double>(1, 0) = images_keypoints[i + 1][match.trainIdx].pt.y;
				p1.at<double>(2, 0) = 1.0;

				Mat diff = transform * p1 - p0;

				if (diff.at<double>(0, 0)* diff.at<double>(0, 0) + diff.at<double>(1, 0) * diff.at<double>(1, 0) < c)
					inlier_count++;
			}

			if (inlier_count > max_inlier_count)
			{
				max_transform = transform.clone();
				max_inlier_count = inlier_count;
				max_matchs = sample_matchs;
			}
		}
		transforms.push_back(max_transform);

		Mat img_matches;
		Mat img0, img1;
		cv::cvtColor(cylindrical_images[i], img0, CV_RGBA2RGB);
		cv::cvtColor(cylindrical_images[i+1], img1, CV_RGBA2RGB);
		drawMatches(
			img0, images_keypoints[i],
			img1, images_keypoints[i + 1],
			max_matchs, img_matches, Scalar::all(-1), CV_RGB(255, 255, 255), Mat(), 4);
		imshow("Mathc", img_matches);
		waitKey(0);
	}
	//accumulate transform matrix
	for (int i = 1; i < transforms.size(); i++)
		transforms[i] = transforms[i - 1] * transforms[i];

	// Fix up the end-to-end alignment
	double y_aligment = transforms.back().at<double>(1, 2);
	for (int i = 0; i < transforms.size(); i++)
	{
		double y_offset = y_aligment * (i / (double)(transforms.size()-1));
		transforms[i].at<double>(1, 2) -= y_offset;
	}

	
	// Blending
	int y_offset = cylindrical_images[0].rows / 2;
	int w = 0;
	int h = cylindrical_images[0].rows * 2;
	for (Mat& image : cylindrical_images)
		w += image.cols;

	Mat stitch = Mat::zeros(cv::Size(w, h), CV_8UC4);
	for (int x = 0; x < cylindrical_images[0].cols; x++)
		for (int y = 0; y < cylindrical_images[0].rows; y++)
		{
			float xx = x;
			float yy = y_offset + y;
			Vec4b color = cylindrical_images[0].at<Vec4b>(y, x);
			if (color[3] == 0)
				continue;
			stitch.at<Vec4b>((int)yy, (int)xx) = Vec4b(color[0], color[1], color[2], 255);
		}
	for (int i = 1; i < transforms.size(); i++)
	{
		for (int x = 0; x < cylindrical_images[i].cols; x++)
			for (int y = 0; y < cylindrical_images[i].rows; y++)
			{
				//float xx = x + translations[i].x;
				//float yy = y_offset + y + translations[i].y;
				Mat pos(cv::Size(1, 3), CV_64FC1);
				pos.at<double>(0, 0) = x;
				pos.at<double>(1, 0) = y;
				pos.at<double>(2, 0) = 1;
				Mat trans_pos = transforms[i] * pos;

				float xx = trans_pos.at<double>(0, 0);
				float yy = y_offset + trans_pos.at<double>(1, 0);

				if (xx >= stitch.cols || yy >= stitch.rows || xx < 0 || yy < 0)
					continue;
				Vec4b src = cylindrical_images[i].at<Vec4b>(y, x);
				if (src[3] == 0)
					continue;
				
				Vec4b dst = stitch.at<Vec4b>((int)yy, (int)xx);
				if (dst[3] == 255)
				{
					//blend
					float left = transforms[i].at<double>(0, 2);
					float right = transforms[i-1].at<double>(0, 2) + cylindrical_images[i-1].cols;
					//float middle = (left + right) * 0.5f;
					//left = middle - 5.0f;
					//right = middle + 5.0f;
					float alpha;
					if (xx < left)
						alpha = 1.0f;
					else if (xx > right)
						alpha = 0.0f;
					else
						alpha = (xx - left) / (right - left);

					//float alpha = 0.5f;
					src[0] = src[0] * alpha + dst[0] * (1.0f - alpha);
					src[1] = src[1] * alpha + dst[1] * (1.0f - alpha);
					src[2] = src[2] * alpha + dst[2] * (1.0f - alpha);
				}
				stitch.at<Vec4b>((int)yy, (int)xx) = Vec4b(src[0], src[1], src[2], 255);



			}
		cv::imwrite("temp" + to_string(i) + ".jpg", stitch);
	}

	// Crop the result and import into a viewer
	int min_x = INT_MAX;
	int min_y = INT_MAX;
	int max_x = 0;
	int max_y = 0;
	for (int x = 0; x < stitch.cols; x++)
		for (int y = 0; y < stitch.rows; y++)
		{
			if (stitch.at<Vec4b>(y, x)[3] > 0)
			{
				if (x < min_x)
					min_x = x;
				if (y < min_y)
					min_y = y;
				if (x > max_x)
					max_x = x;
				if (y > max_y)
					max_y = y;
			}
		}

	Mat result(cv::Size(max_x - min_x + 1, max_y - min_y + 1), CV_8UC3);
	for (int x = 0; x < result.cols; x++)
		for (int y = 0; y < result.rows; y++)
		{
			result.at<Vec3b>(y, x)[0] = stitch.at<Vec4b>(y + min_y, x + min_x)[0];
			result.at<Vec3b>(y, x)[1] = stitch.at<Vec4b>(y + min_y, x + min_x)[1];
			result.at<Vec3b>(y, x)[2] = stitch.at<Vec4b>(y + min_y, x + min_x)[2];
		}

	return result;
}