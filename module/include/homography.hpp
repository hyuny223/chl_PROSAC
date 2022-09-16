#pragma once

#ifndef MY_HOMOGRAPHY_HPP
#define MY_HOMOGRAPHY_HPP

#include <iostream>
#include <vector>
#include <random>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SVD"

#include "RANSAC.hpp"
#include "detect.hpp"

class Homography : public DetectMatch
{
protected:
    Eigen::MatrixXd B_;
    Eigen::MatrixXd C_;
    Eigen::MatrixXd curr_, next_, pred_;
    std::vector<std::vector<cv::DMatch>> sorted_;
    std::vector<cv::KeyPoint> keys1_, keys2_;

    double error_;

public:
    Homography() = default;
    void run(const std::vector<std::vector<cv::DMatch>>& sorted,
             const std::vector<cv::KeyPoint>& keys1,
             const std::vector<cv::KeyPoint>& keys2);
    void computeSVD();
    void computeB();
    void computeC(const Eigen::JacobiSVD<Eigen::MatrixXd>& svd);
    void compute2D();
    void computeRMSE();
    double getError();
    Eigen::MatrixXd &getHomography();
};

#endif
