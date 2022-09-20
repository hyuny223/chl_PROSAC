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
    std::vector<cv::Point2f> left_, right_;
    std::vector<cv::KeyPoint> keys1_, keys2_;
    std::vector<int> indices_;

    double res_th_{10.0};
    int n_;
public:
    double error_;
    int inliers_{0};

public:
    Homography() = default;
    void set(const std::vector<std::vector<cv::DMatch>>& sorted,
             const std::vector<cv::KeyPoint> &keys1,
             const std::vector<cv::KeyPoint> &keys2);
    bool check(int n);
    void sampling(std::mt19937 &gen);
    void run(int n, std::mt19937 &gen);
    void computeSVD();
    void computeB();
    void computeC(const Eigen::JacobiSVD<Eigen::MatrixXd>& svd);
    void compute2D();
    void computeInliers();
    void computeRMSE();
    Eigen::MatrixXd &getHomography();
};

#endif
