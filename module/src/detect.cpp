#include <iostream>
#include <tuple>

#include "opencv2/opencv.hpp"
#include "easy/profiler.h"

#include "detect.hpp"
#include "chrono"

DetectMatch::DetectMatch(const std::string &path)
{
    EASY_FUNCTION(profiler::colors::Magenta);
    std::string c = path + "left.png";
    std::string n = path + "right.png";

    curr_ = cv::imread(c, cv::IMREAD_GRAYSCALE);
    next_ = cv::imread(n, cv::IMREAD_GRAYSCALE);
}

void DetectMatch::detectFeatures()
{
    EASY_FUNCTION(profiler::colors::Magenta);
    cv::Ptr<cv::Feature2D> detector;
    if (mode_ == "SIFT")
    {
        detector = cv::SIFT::create(feat_num_);
    }
    else if (mode_ == "ORB")
    {
        detector = cv::ORB::create();
    }

    cv::Mat mask;
    EASY_BLOCK("detectAndCompute", profiler::colors::Blue500);
    detector->detectAndCompute(curr_, mask, keys1_, desc1_);
    detector->detectAndCompute(next_, mask, keys2_, desc2_);
}

void DetectMatch::match()
{
    EASY_FUNCTION(profiler::colors::Magenta);
    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::KDTreeIndexParams>(5));
    matcher.knnMatch(desc1_, desc2_, matches_, 2);
}

void DetectMatch::show(Eigen::MatrixXd H)
{
    cv::Mat h = (cv::Mat_<double>(3, 3) << H(0, 0), H(0, 1), H(0, 2),
                 H(1, 0), H(1, 1), H(1, 2),
                 H(2, 0), H(2, 1), H(2, 2));
    cv::Mat dst, tmp1;

    auto s = std::chrono::steady_clock::now();
    EASY_BLOCK("opencv", profiler::colors::Blue500);
    std::vector<cv::Point2f> k1, k2;
    std::vector<cv::DMatch> good;
    for (int i = 0; i < matches_.size(); ++i)
    {
        if (matches_[i][0].distance < matches_[i][1].distance * 0.2)
        {
            good.push_back(matches_[i][0]);
        }
    }

    for (auto g : good)
    {
        k1.push_back(keys1_[g.queryIdx].pt);
        k2.push_back(keys2_[g.trainIdx].pt);
    }
    
    auto tmp = cv::findHomography(k1, k2);
    Eigen::MatrixXd u(3, 3);
    u << tmp.ptr<double>(0)[0], tmp.ptr<double>(0)[1], tmp.ptr<double>(0)[2],
         tmp.ptr<double>(1)[0], tmp.ptr<double>(1)[1], tmp.ptr<double>(1)[2],
         tmp.ptr<double>(2)[0], tmp.ptr<double>(2)[1], tmp.ptr<double>(2)[2];

    Eigen::MatrixXd a(3, k1.size());
    for (int i = 0; i < k1.size(); ++i)
    {
        Eigen::MatrixXd c(3, 1);
        c << k1[i].x, k1[i].y, 1.0;

        a.block<3, 1>(0, i) = c;
    }
    auto T = (u * a);                              // col이 3인 형태 (x, y, a)
    auto pred = T.array().rowwise() / T.row(2).array(); // (x/a, y/a, 1)

    double error{0};
    for (int i = 0; i < k1.size(); ++i)
    {
        auto p_x = static_cast<float>(pred(0, i));
        auto p_y = static_cast<float>(pred(1, i));
        auto n_x = k2[i].x;
        auto n_y = k2[i].y;

        double err = std::sqrt((n_x - p_x) * (n_x - p_x) + (n_y - p_y) * (n_y - p_y));
        error += err;
    }

    auto e = std::chrono::steady_clock::now();
    std::chrono::duration<double> t = e - s;
    std::cout << "opencv time : " << t.count() << std::endl;
    std::cout << "opencv error : " << error << std::endl;
    EASY_END_BLOCK;
    cv::warpPerspective(curr_, dst, h, curr_.size(), cv::INTER_LANCZOS4);
    cv::warpPerspective(curr_, tmp1, tmp, curr_.size(), cv::INTER_LANCZOS4);

    cv::imshow("left", curr_);
    cv::imshow("right", next_);
    cv::imshow("result", dst);
    cv::imshow("tmp", tmp1);
    while (1)
    {
        if (cv::waitKey(0) == 27)
        {
            break;
        }
    }
}
