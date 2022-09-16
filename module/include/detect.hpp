#pragma once

#ifndef MY_DETECT_CPP
#define MY_DETECT_HPP

#include <iostream>
#include <tuple>

#include "opencv2/opencv.hpp"
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SVD"

class DetectMatch
{
protected:
    cv::Mat curr_, next_;
    std::string mode_ = "SIFT";
    int feat_num_ = 500;

    std::vector<cv::KeyPoint> keys1_, keys2_;
    cv::Mat desc1_, desc2_;

    std::vector<std::vector<cv::DMatch>> matches_;

public:
    DetectMatch() = default;
    DetectMatch(const std::string &path);
    void detectFeatures();
    void match();
    void show(Eigen::MatrixXd H);
    auto run()
    {
        detectFeatures();
        match();

        return std::tuple{matches_, keys1_, keys2_};
    }
};

#endif
