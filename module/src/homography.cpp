#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SVD"
#include "easy/profiler.h"

#include "homography.hpp"

void Homography::set(const std::vector<std::vector<cv::DMatch>> &sorted,
                     const std::vector<cv::KeyPoint> &keys1,
                     const std::vector<cv::KeyPoint> &keys2)
{
    EASY_FUNCTION(profiler::colors::Magenta);
    sorted_ = sorted;
    keys1_ = keys1;
    keys2_ = keys2;

    left_.clear();
    right_.clear();
    left_.reserve(keys1.size());
    right_.reserve(keys2.size());

    for (const auto &m : sorted)
    {
        left_.push_back(keys1[m[0].queryIdx].pt);
        right_.push_back(keys2[m[0].trainIdx].pt);
    }
}

bool Homography::check(int n)
{
    for (auto i : indices_)
    {
        if (i == n)
        {
            return false;
        }
    }
    return true;
}

void Homography::sampling(std::mt19937 &gen)
{
    EASY_FUNCTION(profiler::colors::Magenta);
    std::uniform_int_distribution<int> dis(0, n_ - 1);
    indices_.clear();
    indices_.reserve(4);

    int cnt = 0;
    EASY_BLOCK("check comb", profiler::colors::Blue500);
    while (cnt < 4)
    {
        int idx = dis(gen);
        if (check(idx))
        {
            indices_.push_back(idx); // 중복 검사 안 해줘도 될까?
            ++cnt;
        }
    }
    EASY_END_BLOCK;
    EASY_BLOCK("sampling key point", profiler::colors::Blue500);
    Eigen::MatrixXd a(3, n_), b(3, n_);
    for (int i = 0; i < n_; ++i)
    {
        Eigen::MatrixXd c(3, 1), n(3, 1);
        c << left_[i].x, left_[i].y, 1;
        n << right_[i].x, right_[i].y, 1;

        a.block<3, 1>(0, i) = c;
        b.block<3, 1>(0, i) = n;
    }
    EASY_END_BLOCK;
    curr_ = a;
    next_ = b;
}

void Homography::run(int n, std::mt19937 &gen)
{
    EASY_FUNCTION(profiler::colors::Magenta);
    n_ = n;
    EASY_BLOCK("sampling", profiler::colors::Blue500);
    sampling(gen);
    EASY_END_BLOCK;
    EASY_BLOCK("svd", profiler::colors::Blue500);
    computeSVD();
    EASY_END_BLOCK;
    EASY_BLOCK("compute inliers", profiler::colors::Blue500);
    computeInliers();
}

void Homography::computeSVD()
{
    EASY_FUNCTION(profiler::colors::Magenta);
    EASY_BLOCK("compute B", profiler::colors::Blue500);
    computeB();
    EASY_END_BLOCK;
    EASY_BLOCK("decomposition", profiler::colors::Blue500);
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(B_, Eigen::ComputeThinU | Eigen::ComputeThinV);
    EASY_END_BLOCK;
    EASY_BLOCK("compute C", profiler::colors::Blue500);
    computeC(svd);
    EASY_END_BLOCK;
    EASY_BLOCK("compute 2D", profiler::colors::Blue500);
    compute2D();
}

void Homography::computeB()
{
    EASY_FUNCTION(profiler::colors::Magenta);
    Eigen::MatrixXd B(8, 9);
    // cv::Mat board(1000, 1000, CV_8UC3, cv::Scalar::all(255));

    int row = 1;
    EASY_BLOCK("making H in compute B", profiler::colors::Blue500);
    for (const int i : indices_)
    {
        cv::Point2d p = left_[i];
        cv::Point2d q = right_[i];
        // cv::circle(board, cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)), 3, cv::Scalar(255, 0, 0), 3);
        // cv::circle(board, cv::Point(static_cast<int>(q.x), static_cast<int>(q.y)), 3, cv::Scalar(0, 0, 255), 3);

        auto x = p.x, y = p.y;
        auto u = q.x, v = q.y;

        Eigen::MatrixXd A1(1, 9), A2(1, 9), P(3, 1), N(1, 3);
        A1 << x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, -u;
        A2 << 0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, -v;
        P << x, y, 1;
        N << u, v, 1;

        B.block<1, 9>(2 * row - 2, 0) = A1;
        B.block<1, 9>(2 * row - 1, 0) = A2;
        ++row;
    }
    EASY_END_BLOCK;
    // cv::imshow("a", board);
    // cv::waitKey(0);
    B_ = B;

}

void Homography::computeC(const Eigen::JacobiSVD<Eigen::MatrixXd> &svd)
{
    EASY_FUNCTION(profiler::colors::Magenta);
    Eigen::MatrixXd C(3, 3);
    auto tmp = svd.matrixV().col(8);

    C.block<1, 3>(0, 0) = tmp.block<3, 1>(0, 0);
    C.block<1, 3>(1, 0) = tmp.block<3, 1>(3, 0);
    C.block<1, 3>(2, 0) = tmp.block<3, 1>(6, 0);

    C_ = C;
}

void Homography::compute2D()
{
    EASY_FUNCTION(profiler::colors::Magenta);
    auto T = (C_ * curr_);                              // col이 3인 형태 (x, y, a)
    auto pred = T.array().rowwise() / T.row(2).array(); // (x/a, y/a, 1)

    pred_ = pred;
}

void Homography::computeInliers()
{
    EASY_FUNCTION(profiler::colors::Magenta);

    int cnt = 0;
    double error{0};

    for (int i = 0; i < n_; ++i)
    {

        auto p_x = pred_(0, i);
        auto p_y = pred_(1, i);
        auto n_x = next_(0, i);
        auto n_y = next_(1, i);

        double e = std::sqrt((n_x - p_x) * (n_x - p_x) + (n_y - p_y) * (n_y - p_y));
        cnt = e < res_th_ ? cnt + 1 : cnt;
        error += e;
    }
    inliers_ = cnt;
    error_ = error / n_;
}

void Homography::computeRMSE()
{
    std::size_t len = sorted_.size();

    double error{0};
    double lowest = std::numeric_limits<double>::lowest();

    for (int i = 0; i < len; ++i)
    {

        auto p_x = pred_(i, 0);
        auto p_y = pred_(i, 1);
        auto n_x = next_(i, 0);
        auto n_y = next_(i, 1);

        double e = std::sqrt((n_x - p_x) * (n_x - p_x) + (n_y - p_y) * (n_y - p_y));
        error += e;                       // prosac 논문에서는 lowest quality of data가 subset의 퀄리티가 된다고 하였는데, 그냥 RMSE로 계산하였다.
        lowest = e > lowest ? e : lowest; // 이게 원래 논문 방식. lowest가 error가 크다는 의미.
    }

    std::cout << "The number of points : " << len << std::endl;
    std::cout << "RMSE : " << error / len << std::endl;
    std::cout << "lowest : " << lowest << std::endl;

    error_ = error / len;
}

Eigen::MatrixXd &Homography::getHomography()
{
    return C_;
}
