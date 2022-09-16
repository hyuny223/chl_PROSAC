#pragma once
#ifndef MY_RANSAC_HPP
#define MY_RANSAC_HPP

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "opencv2/opencv.hpp"

#include "detect.hpp"

/*
    N(N_): Total number of samples. (fixed)

    m(m_): Complexity of the geometric model.

    t(iteration_): Number of iterations of the loop. (o)

    n*(ns_): Termination length (number of samples to consider before stopping).

    TN(TN_): Parameter that defines after how many samples PROSAC becomes equivalent to RANSAC. (o)

    Tn(Tn_): Average number of samples from the n top-ranked points.

    Tn'(Tnf_): Ceiled value of Tn (o)
*/

template <typename MODEL>
class PROSAC
{
protected:
    MODEL model_, candidate_;

    std::vector<std::vector<cv::DMatch>> sorted_, match_;
    std::vector<cv::KeyPoint> keys1_, keys2_;

    double best_model_err_ = std::numeric_limits<double>::infinity();
    double candidate_model_err_ = std::numeric_limits<double>::infinity();

    int iteration_;
    int N_, m_, ns_, TN_, Tn_;

    double threshold_;

public:
    // PROSAC() {};
    PROSAC(std::vector<std::vector<cv::DMatch>> matches,
           const std::vector<cv::KeyPoint> &keys1,
           const std::vector<cv::KeyPoint> &keys2,
           const int &iteration,
           const double &threshold,
           MODEL &model)
        : keys1_(keys1), keys2_(keys2), iteration_(iteration), threshold_(threshold), model_(model)
    {
        N_ = matches.size();
        m_ = 10;
        std::sort(matches.begin(), matches.end(), [](auto i, auto j){ return i[0].distance / i[1].distance < j[0].distance / j[1].distance ? 1 : 0; });
        sorted_ = matches;
    }

    MODEL run()
    {
        for (int t = 0; t < iteration_; ++t)
        {
            makeSubset(m_);
            if (iterate())
            {
                if (best_model_err_ < threshold_)
                {
                    std::cout << "model!" << std::endl;
                    return model_;
                }
                if (model_.getError() < candidate_model_err_)
                {
                    candidate_ = model_;
                }
            }
            ++m_;
        }
        std::cout << "candidate!" << std::endl;
        return candidate_;
    }
    bool iterate()
    {
        model_.run(match_, keys1_, keys2_);
        auto err = model_.getError();

        if (err < best_model_err_)
        {
            best_model_err_ = err;
            return true;
        }
        return false;
    }

    void makeSubset(const int &m)
    {
        // 최소 5개부터는 시작해야 호모그래피를 찾을 수 있음
        // 지금은 슬라이싱으로 하고 있는데, 기존 벡터에 뒤에 하나만 덫붙이는 방식으로 가능할 듯
        std::vector<std::vector<cv::DMatch>> match(sorted_.begin(), sorted_.begin() + m);
        match_ = match;
    }

    MODEL &getModel()
    {
        return model_;
    }
};

#endif
