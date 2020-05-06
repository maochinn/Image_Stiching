#include "FeatureDetector.h"

void drawFeatures(const std::vector<FeaturePoint>& features, cv::Mat& src) {
    cv::Mat copy;
    src.copyTo(copy);
    for (auto feature : features) {
        cv::circle(copy, cv::Point(feature.x, feature.y), 3, cv::Scalar(0, 0, 255));
    }
    cv::imshow("features.jpg", copy);
    cv::waitKey(1);
}

cv::Mat DetectFeature(cv::Mat src, std::vector<FeaturePoint>& features, int level, int scale, float feature_threshold, int max_feature, int non_max_r, int non_max_step, int suppression_mode)
{
    float sigma = 1.0;
    float sigma_d = 1.0;
    float sigma_i = 1.5;
    float sigma_o = 4.5;

	cv::Mat gray;
	cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
	std::vector<cv::Mat> pyramid(level);
    for (int i = 0; i < level; i++) {
        gray.copyTo(pyramid[i]);
    }
	getPyramid(pyramid, scale, level, sigma);

    // Harris corner response
    std::vector<cv::Mat> response(level);
    for (int i = 0; i < level; i++) {
        pyramid[i].copyTo(response[i]);
    }
    HarrisResponse(pyramid, response, level, sigma_d, sigma_i);

    // feature orientation
    std::vector<cv::Mat> orientation(level);
    for (int i = 0; i < level; i++) {
        pyramid[i].copyTo(orientation[i]);
    }
    featureOrientation(pyramid, orientation, level, sigma_o);

    // find local maxima and thresholding
    bool** isFeature = new bool* [level];
    for (int lv = 0; lv < level; lv++)
        isFeature[lv] = new bool[response[lv].cols * response[lv].rows];
    findFeatures(response, isFeature, level, feature_threshold);

    // sub-pixel accuracy
    subPixelAccuracy(response, isFeature, level);

    // project features to original resolution
    FeaturePoint** featureMap = new FeaturePoint * [src.rows];
    for (int i = 0; i < src.rows; i++)
        featureMap[i] = new FeaturePoint[src.cols];
    projectFeatures(features, featureMap, response, orientation, isFeature, level, scale);
    response.resize(0);
    orientation.resize(0);
    drawFeatures(features, src);

    int* counter = new int[level];
    for (int lv = 0; lv < level; lv++)
        counter[lv] = 0;
    for (int k = 0; k < features.size(); k++)
        counter[features[k].level]++;
    for (int lv = 0; lv < level; lv++)
        std::cout << "level " << lv << ": " << counter[lv] << std::endl;

    // delete features too close to the boundaries
    deleteCloseToBounds(features, pyramid, level, scale);
    drawFeatures(features, src);
    std::cout << "\tNumber of Features: " << features.size() << std::endl;
    for (int lv = 0; lv < level; lv++)
        counter[lv] = 0;
    for (int k = 0; k < features.size(); k++)
        counter[features[k].level]++;
    for (int lv = 0; lv < level; lv++)
        std::cout << "level " << lv << ": " << counter[lv] << std::endl;

    if (suppression_mode == 0) {
        // non-maximal suppression
        std::cout << "\nApply non-maximal suppression ...\n" << std::endl;
        nonMaximalSuppression(features, max_feature, non_max_r, non_max_step);
    }
    else {
        // strongest
        std::cout << "\nApply strongest n ...\n" << std::endl;
        strongest(features, max_feature);
    }
    drawFeatures(features, src);
    std::cout << "\tNumber of Features: " << features.size() << std::endl;
    for (int lv = 0; lv < level; lv++)
        counter[lv] = 0;
    for (int k = 0; k < features.size(); k++)
        counter[features[k].level]++;
    for (int lv = 0; lv < level; lv++)
        std::cout << "level " << lv << ": " << counter[lv] << std::endl;

    // feature descriptor
    std::cout << "\nCompute feature descriptors ..." << std::endl;
    featureDescriptor(features, pyramid, level, scale);

    for (int i = 0; i < src.rows; i++)
        delete[] featureMap[i];
    delete[] featureMap;
    delete[] counter;
	return src;
}

void getPyramid(std::vector<cv::Mat>& pyramid, int _scale, int _level, float _sigma)
{
    for (int lv = _level - 2; lv >= 0; lv--) {
        // gaussian blur
        cv::GaussianBlur(pyramid[lv + 1], pyramid[lv], cv::Size(3, 3), _sigma);
        // downsample
        cv::Size dsize = cv::Size(pyramid[lv + 1].cols / _scale, pyramid[lv + 1].rows / _scale);
        cv::resize(pyramid[lv + 1], pyramid[lv], dsize);
    }
}

void computeGradient(cv::Mat src, cv::Mat& dst, int xOrder, int yOrder)
{
    for (int y = 0; y < dst.rows; y++) {
        for (int x = 0; x < dst.cols; x++) {
            float tmp1 = 0.0, tmp2 = 0.0;

            /*first order gradient*/
            if ((xOrder == 1) && (yOrder == 0)) {
                if (x - 1 >= 0)
                    tmp1 = src.at<uchar>(y, x - 1);
                if (x + 1 < dst.cols)
                    tmp2 = src.at<uchar>(y, x + 1);
                dst.at<uchar>(y, x) = (tmp2 - tmp1) / 2;
            }

            else if ((xOrder == 0) && (yOrder == 1)) {
                if (y - 1 >= 0)
                    tmp1 = src.at<uchar>(y - 1, x);
                if (y + 1 < dst.rows)
                    tmp2 = src.at<uchar>(y + 1, x);
                dst.at<uchar>(y, x) = (tmp2 - tmp1) / 2;
            }
            
            /*second order gradient*/
            else if ((xOrder == 1) && (yOrder == 1)) {
                if ((x - 1 >= 0) && (y - 1 >= 0))
                    tmp2 = src.at<uchar>(y - 1, x - 1);
                if ((x + 1 < dst.cols) && (y + 1 < dst.rows))
                    tmp2 += src.at<uchar>(y + 1, x + 1);
                if ((x - 1 >= 0) && (y + 1 < dst.rows))
                    tmp1 = src.at<uchar>(y + 1, x - 1);
                if ((x + 1 < dst.cols) && (y - 1 >= 0))
                    tmp1 += src.at<uchar>(y - 1, x + 1);
                dst.at<uchar>(y, x) = (tmp2 - tmp1) / 4;
            }
            else if ((xOrder == 0) && (yOrder == 2)) {
                if (y - 1 >= 0)
                    tmp1 = src.at<uchar>(y - 1, x);
                if (y + 1 < dst.rows)
                    tmp2 = src.at<uchar>(y + 1, x);
                dst.at<uchar>(y, x) = tmp2 - 2 * src.at<uchar>(y, x) + tmp1;
            }
            else if ((xOrder == 2) && (yOrder == 0)) {
                if (x - 1 >= 0)
                    tmp1 = src.at<uchar>(y, x - 1);
                if (x + 1 < dst.cols)
                    tmp2 = src.at<uchar>(y, x + 1);
                dst.at<uchar>(y, x) = tmp2 - 2 * src.at<uchar>(y, x) + tmp1;
            }
        }
    }
}

void HarrisResponse(std::vector<cv::Mat> pyramid, std::vector<cv::Mat>& response, int _level, float _sigma_d, float _sigma_i)
{
    cv::Mat Ix, Iy, Ix2, Iy2, IxIy, img;
    for (int lv = 0; lv < _level; lv++) {
        pyramid[lv].copyTo(img);
        pyramid[lv].copyTo(Ix);
        pyramid[lv].copyTo(Iy);

        // compute Ix, Iy of the image at each level of the pyramid
        computeGradient(pyramid[lv], img, 1, 0);
        cv::GaussianBlur(img, Ix, cv::Size(3, 3), _sigma_d);
        computeGradient(pyramid[lv], img, 0, 1);
        cv::GaussianBlur(img, Iy, cv::Size(3, 3), _sigma_d);
        /*cv::imshow("gx" + std::to_string(lv + 1) + ".jpg", Ix);
        cv::waitKey(1);*/
        /*cv::imshow("gy" + std::to_string(lv + 1) + ".jpg", Iy);
        cv::waitKey(1);*/


        // compute Ix2, Iy2, and Ixy (product of derivatives) at each level of the pyramid
        pyramid[lv].copyTo(Ix2);
        pyramid[lv].copyTo(Iy2);
        pyramid[lv].copyTo(IxIy);
        // second order gradients and blur -> Ix2, Iy2, Ixy
        for (int i = 0; i < pyramid[lv].rows; i++)
            for (int j = 0; j < pyramid[lv].cols; j++)
                img.at<uchar>(i, j) = Ix.at<uchar>(i, j) * Ix.at<uchar>(i, j);
        cv::GaussianBlur(img, Ix2, cv::Size(3, 3), _sigma_i);
        for (int i = 0; i < pyramid[lv].rows; i++)
            for (int j = 0; j < pyramid[lv].cols; j++)
                img.at<uchar>(i, j) = Iy.at<uchar>(i, j) * Iy.at<uchar>(i, j);
        cv::GaussianBlur(img, Iy2, cv::Size(3, 3), _sigma_i);
        for (int i = 0; i < pyramid[lv].rows; i++)
            for (int j = 0; j < pyramid[lv].cols; j++)
                img.at<uchar>(i, j) = Ix.at<uchar>(i, j) * Iy.at<uchar>(i, j);
        cv::GaussianBlur(img, IxIy, cv::Size(3, 3), _sigma_i);

        // compute Harris corner response
        // M = [ Ix2   IxIy ]
        //     [ IxIy  Iy2  ]
        // det(M) = |M| = Ix2 * Iy2 - IxIy * IxIy
        // tr(M) = sum of diagonal = Ix2 + Iy2
        for (int i = 0; i < response[lv].rows; i++) {
            for (int j = 0; j < response[lv].cols; j++) {
                response[lv].at<uchar>(i, j) = 255 * 255;
                
                if (Ix2.at<uchar>(i, j) + Iy2.at<uchar>(i, j) == 0.0)
                    response[lv].at<uchar>(i, j) = 0.0;
                else
                    response[lv].at<uchar>(i, j) *= ((Ix2.at<uchar>(i, j) * Iy2.at<uchar>(i, j)) - (IxIy.at<uchar>(i, j) * IxIy.at<uchar>(i, j))) / (Ix2.at<uchar>(i, j) + Iy2.at<uchar>(i, j));
            }
        }
        /*cv::imshow("resp" + std::to_string(lv + 1) + ".jpg", response[lv]);
        cv::waitKey(1);*/
    }
}

void featureOrientation(const std::vector<cv::Mat>& pyramid, std::vector<cv::Mat>& orientation, int _level, float _sigma_o)
{
    cv::Mat Ix, Iy, img;
    for (int lv = 0; lv < _level; lv++) {
        pyramid[lv].copyTo(img);
        pyramid[lv].copyTo(Ix);
        pyramid[lv].copyTo(Iy);
        // compute Ix, Iy of the image at each level of the pyramid
        computeGradient(pyramid[lv], img, 1, 0);
        cv::GaussianBlur(img, Ix, cv::Size(3, 3), _sigma_o);
        computeGradient(pyramid[lv], img, 0, 1);
        cv::GaussianBlur(img, Iy, cv::Size(3, 3), _sigma_o);

        // [cos(theta), sin(theta)] = [Ix, Iy] 
        // => theta = atan(Iy / Ix)
        for (int i = 0; i < orientation[lv].rows; i++)
            for (int j = 0; j < orientation[lv].cols; j++)
                orientation[lv].at<char>(i, j) = atan2(Iy.at<char>(i, j), Ix.at<char>(i, j));
    }
}

void findFeatures(const std::vector<cv::Mat>& response, bool** isFeature, int level, float threshold)
{
    for (int lv = 0; lv < level; lv++) {
        int w = response[lv].cols;
        int h = response[lv].rows;

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                isFeature[lv][i * w + j] = false;
                // if responce greater than the threshold then it may be a feature
                if (response[lv].at<char>(i, j) > threshold) {
                    isFeature[lv][i * w + j] = true;

                    // check if (i, j) is the maxima point in the 3 x 3 region
                    for (int u = -1; u <= 1 && isFeature[lv][i * w + j]; u++) {
                        if ((i + u < 0) || (i + u >= h))
                            continue;
                        for (int v = -1; v <= 1; v++) {
                            if ((j + v < 0) || (j + v >= w))
                                continue;
                            if (response[lv].at<char>(i + u, j + v) > response[lv].at<char>(i, j)) {
                                isFeature[lv][i * w + j] = false;
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
}

void subPixelAccuracy(const std::vector<cv::Mat>& res, bool** isFeature, int level)
{
    cv::Mat Ix, Iy, Ix2, Iy2, Ixy;
    // work, work
    for (int lv = 0; lv < level; lv++) {
        int w = res[lv].cols;
        int h = res[lv].rows;

        res[lv].copyTo(Ix);
        res[lv].copyTo(Iy);
        res[lv].copyTo(Ix2);
        res[lv].copyTo(Iy2);
        res[lv].copyTo(Ixy);

        computeGradient(res[lv], Ix, 1, 0);
        computeGradient(res[lv], Iy, 0, 1);
        computeGradient(res[lv], Ix2, 2, 0);
        computeGradient(res[lv], Iy2, 0, 2);
        computeGradient(res[lv], Ixy, 1, 1);

        /* parse isFeature */
        std::vector<FeaturePoint> Pts;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                if (isFeature[lv][i * w + j]) {
                    FeaturePoint Pt;
                    Pt.x = j;
                    Pt.y = i;
                    Pts.push_back(Pt);
                }
            }
        }
        /* parse done */

        // Xm = -[A]inverse * [B]  , A=d2f/d<x>2, B=df/d<x>
        for (int i = 0; i < Pts.size(); i++) {
            float A[2][2] = { { Ix2.at<uchar>(Pts[i].y, Pts[i].x), Ixy.at<uchar>(Pts[i].y, Pts[i].x) },
                               { Ixy.at<uchar>(Pts[i].y, Pts[i].x), Iy2.at<uchar>(Pts[i].y, Pts[i].x) } }; // A
            float detA = A[0][0] * A[1][1] - A[0][1] * A[1][0]; // det(A)
            float Ai[2][2] = { {  A[1][1] / detA, -A[0][1] / detA },
                               { -A[1][0] / detA,  A[0][0] / detA } }; // A inverse
            float B[2] = { Ix.at<uchar>(Pts[i].y, Pts[i].x), Iy.at<uchar>(Pts[i].y, Pts[i].x) }; // B
            float offset[2] = { -Ai[0][0] * B[0] - Ai[0][1] * B[1], -Ai[1][0] * B[0] - Ai[1][1] * B[1] }; // ans
            // if the offset if larger than 0.5, shift the sample point of the feature once
            if (offset[0] > 0.5 || offset[0] < -0.5 || offset[1]>0.5 || offset[1] < -0.5) {
                /*make shift to isFeature map*/
                isFeature[lv][Pts[i].x + Pts[i].y * w] = false;
                if (offset[0] > 0.5 && Pts[i].x + 1 < w)
                    Pts[i].x++;
                else if (offset[0] < -0.5 && Pts[i].x - 1 >= 0)
                    Pts[i].x--;
                if (offset[1] > 0.5 && Pts[i].y + 1 < h)
                    Pts[i].y++;
                else if (offset[1] < -0.5 && Pts[i].y - 1 >= 0)
                    Pts[i].y--;
                isFeature[lv][Pts[i].x + Pts[i].y * w] = true;
            }
        }
    }
}

void projectFeatures(std::vector<FeaturePoint>& features, FeaturePoint** featureMap, const std::vector<cv::Mat>& response, const std::vector<cv::Mat>& orientation, bool** isFeature, int _level, int _scale)
{
    // initialize
    for (int i = 0; i < response[_level - 1].rows; i++) {
        for (int j = 0; j < response[_level - 1].cols; j++) {
            featureMap[i][j].level = -1;
        }
    }

    // project features on all levels 
    for (int lv = _level - 1, s = 1; lv >= 0; lv--, s *= _scale) {
        int w = response[lv].cols;
        int h = response[lv].rows;
        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                // (i, j) on level is a feature
                if (isFeature[lv][i * w + j]) {
                    // (i*s, j*s) on level has no features projected to
                    if (featureMap[i * s][j * s].level == -1) {
                        featureMap[i * s][j * s].x = j * s;
                        featureMap[i * s][j * s].y = i * s;
                        featureMap[i * s][j * s].level = lv;
                        featureMap[i * s][j * s].orientation = orientation[lv].at<uchar>(i, j);
                        featureMap[i * s][j * s].response = response[lv].at<uchar>(i, j);
                    }
                    // (i*s, j*s) on level already has a features projected to
                    else {
                        // if features of different scales project to the same pixel, preserve the one with largest response
                        if (response[lv].at<uchar>(i, j) > featureMap[i * s][j * s].response) {
                            featureMap[i * s][j * s].level = lv;
                            featureMap[i * s][j * s].orientation = orientation[lv].at<uchar>(i, j);
                            featureMap[i * s][j * s].response = response[lv].at<uchar>(i, j);
                        }
                    }
                }
            }
        }
    }

    // all projected features
    for (int i = 0; i < response[_level - 1].rows; i++)
        for (int j = 0; j < response[_level - 1].cols; j++)
            if (featureMap[i][j].level != -1)
                features.push_back(featureMap[i][j]);
}

void deleteCloseToBounds(std::vector<FeaturePoint>& features, const std::vector<cv::Mat>& pyramid, int _level, int _scale)
{
    // initialize
    int* s = new int[_level];
    s[_level - 1] = 1;
    for (int lv = _level - 2; lv >= 0; lv--)
        s[lv] = s[lv + 1] * _scale;

    for (int k = 0; k < features.size(); k++) {
        FeaturePoint fp = features[k];
        // rotate a 40 x 40 descriptor sampling window
        float rotation[2][2] = { {cos(fp.orientation), -sin(fp.orientation)},
                                {sin(fp.orientation),  cos(fp.orientation)} };
        float corner[4][2] = { { rotation[0][0] * (-20) + rotation[0][1] * (-20),
                                rotation[1][0] * (-20) + rotation[1][1] * (-20) }, // top left
                              { rotation[0][0] * (19) + rotation[0][1] * (-20),
                                rotation[1][0] * (19) + rotation[1][1] * (-20) }, // top right
                              { rotation[0][0] * (-20) + rotation[0][1] * (19),
                                rotation[1][0] * (-20) + rotation[1][1] * (19) }, // bottom left
                              { rotation[0][0] * (19) + rotation[0][1] * (19),
                                rotation[1][0] * (19) + rotation[1][1] * (19) } }; // bottom right
        
        // if part of the a descriptor window falls out of image, delete it
        int x = (int)((float)fp.x / s[fp.level]);
        int y = (int)((float)fp.y / s[fp.level]);
        for (int i = 0; i < 4; i++) {
            if ((x + corner[i][0] < 0) 
                || (y + corner[i][1] < 0)
                || (x + corner[i][0] >= pyramid[fp.level].cols)
                || (y + corner[i][1] >= pyramid[fp.level].rows)) {
                features.erase(features.begin() + k);
                k--;
                break;
            }
        }
    }
    delete[] s;
}

inline int featureDistance(FeaturePoint A, FeaturePoint B)
{
    return ((A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y));
}

void nonMaximalSuppression(std::vector<FeaturePoint>& features, int desiredNum, int initRadius, int step)
{
    int desiredNumFixed = (int)(desiredNum + TOLLERATE_RATIO * desiredNum);
    int currentNum = desiredNumFixed + 1;
    std::vector<bool> valid;
    valid.assign(features.size(), true);
    
    for (int radius = initRadius; currentNum > desiredNumFixed; radius += step) {
        int radiusSquared = radius * radius;
        valid.assign(features.size(), true);
        for (int i = 0; i < features.size(); i++) {
            if (!valid[i])
                continue;
            for (int j = 0; j < features.size(); j++) {
                if ((i == j) || featureDistance(features[i], features[j]) >= radiusSquared)
                    continue;
                
                // find strongest feature
                if (features[j].response < features[i].response)
                    valid[j] = false;
                else {
                    valid[i] = false;
                    break;
                }
            }
        }
        currentNum = (int)count(valid.begin(), valid.end(), true);
    }

    // delete features
    for (int i = 0; i < features.size(); i++) {
        if (!valid[i]) {
            features.erase(features.begin() + i);
            valid.erase(valid.begin() + i);
            i--;
        }
    }
}

bool compareByResponse(FeaturePoint a, FeaturePoint b)
{
    return a.response > b.response;
}

void strongest(std::vector<FeaturePoint>& features, int desiredNum)
{
    std::sort(features.begin(), features.end(), compareByResponse);
    features.resize(desiredNum);
}

void featureDescriptor(std::vector<FeaturePoint>& features, const std::vector<cv::Mat>& pyramid, int _level, int _scale)
{
    // initialize
    int* s = new int[_level];
    s[_level - 1] = 1;
    for (int lv = _level - 2; lv >= 0; lv--)
        s[lv] = s[lv + 1] * _scale;
    
    for (int n = 0; n < features.size(); n++) {
        FeaturePoint fp = features[n];
        // rotate a 40 x 40 descriptor sample window
        float rotation[2][2] = { {cos(fp.orientation), -sin(fp.orientation)},
                                {sin(fp.orientation),  cos(fp.orientation)} };
        float window[40][40];
        for (int u = -20; u < 20; u++) {  // row
            for (int v = -20; v < 20; v++) {  // col
                int v_r = (int)(rotation[0][0] * u + rotation[0][1] * v);
                int u_r = (int)(rotation[1][0] * u + rotation[1][1] * v);
                window[u + 20][v + 20] =
                    pyramid[fp.level].at<uchar>((int)(((float)fp.y / s[fp.level]) + u_r), (int)(((float)fp.x / s[fp.level]) + v_r));
            }
        }

        // each element of 64D desrciptor is the mean of every 5 x 5 samples in the 40 x 40 window
        for (int i = 0; i < 40; i += 5) {  // row
            for (int j = 0; j < 40; j += 5) {  // col
                features[n].descriptor[(i / 5) * 8 + (j / 5)] = 0.0;
                for (int u = 0; u < 5; u++)  // row
                    for (int v = 0; v < 5; v++)  // col
                        features[n].descriptor[(i / 5) * 8 + (j / 5)] += window[i + u][j + v];
                features[n].descriptor[(i / 5) * 8 + (j / 5)] /= 25;
            }
        }

        // normalize
        float mean = 0.0;
        for (int i = 0; i < 64; i++)
            mean += features[n].descriptor[i];
        mean /= 64;
        float std = 0.0;
        for (int i = 0; i < 64; i++)
            std += ((features[n].descriptor[i] - mean) * (features[n].descriptor[i] - mean));
        std = sqrt(std / 64);
        for (int i = 0; i < 64; i++)
            features[n].descriptor[i] = (features[n].descriptor[i] - mean) / std;
    }
}



