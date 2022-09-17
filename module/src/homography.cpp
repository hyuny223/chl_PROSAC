#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <random>

#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/SVD"

#include "homography.hpp"

void Homography::run(const std::vector<std::vector<cv::DMatch>> &sorted,
                     const std::vector<cv::KeyPoint> &keys1,
                     const std::vector<cv::KeyPoint> &keys2)
{
    sorted_ = sorted;
    keys1_ = keys1;
    keys2_ = keys2;
    computeSVD();
    computeRMSE();
}

void Homography::computeSVD()
{
    computeB();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B_, Eigen::ComputeFullU | Eigen::ComputeFullV);
    computeC(svd);
    compute2D();
}

void Homography::computeB()
{
    Eigen::MatrixXd B(sorted_.size() * 2, 9);
    Eigen::MatrixXd curr(3, sorted_.size());
    Eigen::MatrixXd next(sorted_.size(), 3);
    cv::Mat board(1000, 1000, CV_8UC3, cv::Scalar::all(255));

    int row = 1;
    for (const auto &m : sorted_)
    {
        cv::Point2d p = keys1_[m[0].queryIdx].pt;
        cv::Point2d q = keys2_[m[0].trainIdx].pt;
        // cv::circle(board, cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)), 3, cv::Scalar(255, 0, 0),3);
        // cv::circle(board, cv::Point(static_cast<int>(q.x), static_cast<int>(q.y)), 3, cv::Scalar(0, 0, 255),3);

        auto x = p.x, y = p.y;
        auto u = q.x, v = q.y;

        Eigen::MatrixXd A1(1, 9), A2(1, 9), P(3, 1), N(1, 3);
        A1 << x, y, 1.0, 0.0, 0.0, 0.0, -x * u, -y * u, -u;
        A2 << 0.0, 0.0, 0.0, x, y, 1.0, -x * v, -y * v, -v;
        P << x, y, 1;
        N << u, v, 1;

        B.block<1, 9>(2 * row - 2, 0) = A1;
        B.block<1, 9>(2 * row - 1, 0) = A2;
        curr.block<3, 1>(0, row - 1) = P;
        next.block<1, 3>(row - 1, 0) = N;
        ++row;
    }
    // cv::imshow("a", board);
    // cv::waitKey(0);
    B_ = B;
    curr_ = curr;
    next_ = next;
}

void Homography::computeC(const Eigen::JacobiSVD<Eigen::MatrixXd> &svd)
{

    Eigen::MatrixXd C(3, 3);
    auto tmp = svd.matrixV().col(8);

    C.block<1, 3>(0, 0) = tmp.block<3, 1>(0, 0);
    C.block<1, 3>(1, 0) = tmp.block<3, 1>(3, 0);
    C.block<1, 3>(2, 0) = tmp.block<3, 1>(6, 0);

    C_ = C;
}

void Homography::compute2D()
{
    auto T = (C_ * curr_).transpose();                  // col이 3인 형태 (x, y, a)
    auto pred = T.array().colwise() / T.col(2).array(); // (x/a, y/a, 1)

    pred_ = pred;
}

void Homography::computeRMSE()
{
    std::size_t len = sorted_.size();

    double error{0};
    double lowest = std::numeric_limits<double>::lowest();

    for (int i = 0; i < len; ++i)
    {
        // auto c_x = static_cast<int>(std::round(gt(i,0)));
        // auto c_y = static_cast<int>(std::round(gt(i,1)));
        // auto n_x = static_cast<int>(std::round(pred(i,0)));
        // auto n_y = static_cast<int>(std::round(pred(i,1)));

        auto p_x = pred_(i, 0);
        auto p_y = pred_(i, 1);
        auto n_x = next_(i, 0);
        auto n_y = next_(i, 1);

        double e = std::sqrt((n_x - p_x) * (n_x - p_x) + (n_y - p_y) * (n_y - p_y));
        error += e; // prosac 논문에서는 lowest quality of data가 subset의 퀄리티가 된다고 하였는데, 그냥 RMSE로 계산하였다.
        lowest = e > lowest? e : lowest; // 이게 원래 논문 방식. lowest가 error가 크다는 의미.
    }

    std::cout << "The number of points : " << len << std::endl;
    std::cout << "RMSE : " << error / len << std::endl;
    std::cout << "lowest : " << lowest << std::endl;

    error_ = error / len;
}

double Homography::getError()
{
    return error_;
}
Eigen::MatrixXd &Homography::getHomography()
{
    return C_;
}
