#pragma once
#include <vector>
struct FeaturePoint {
    int x, y;       // x: col, y: row
    int level;
    float orientation;
    float response;
    float descriptor[64];
    std::vector<int> bestMatch;  // best match in each input image
	std::vector<float> bestMatchDistance;
};