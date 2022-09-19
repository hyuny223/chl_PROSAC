#pragma once
#ifndef MY_RANSAC_HPP
#define MY_RANSAC_HPP

#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

#include "opencv2/opencv.hpp"
#include "easy/profiler.h"

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

    std::vector<std::vector<cv::DMatch>> sorted_, n_match_, m_match_;
    std::vector<cv::KeyPoint> keys1_, keys2_;

    double best_model_inliers_ = std::numeric_limits<double>::lowest();
    double candidate_model_inliers_ = std::numeric_limits<double>::lowest();

    int iteration_;
    int N_, n_, m_, ns_, TN_;

    double threshold_, Tn_, Tpn_;

    double eta_ = 0.05, psi_ = 0.05, beta_ = 0.1;
    double max_outlier_ = 0.6;
    double p_ = 0.99;

public:
    // PROSAC() {};
    PROSAC(std::vector<std::vector<cv::DMatch>> matches,
           const std::vector<cv::KeyPoint> &keys1,
           const std::vector<cv::KeyPoint> &keys2,
           const int &iteration,
           const double &threshold,
           MODEL &model)
        : keys1_(keys1), keys2_(keys2), iteration_(iteration), model_(model)
    {
        EASY_BLOCK("PROSAC INITIALIZE", profiler::colors::Blue500);
        N_ = matches.size();
        threshold_ = N_ * threshold;
        n_ = 4;
        m_ = 4;
        Tn_ = 1.0;
        Tpn_ = 1.0;

        // 퀄리티가 좋은게 뒤로 가게 함으로써 vector에서 pop하기 좋게 만든다.
        // deque으로 하는게 좋은지, 선입선출, 후입선출 뭐가 좋은지 전부 수행해보면 좋을 듯.
        EASY_BLOCK("sorting", profiler::colors::Blue500);
        std::sort(matches.begin(), matches.end(), [](auto i, auto j)
                  { return i[0].distance / i[1].distance < j[0].distance / j[1].distance ? 1 : 0; });
        EASY_END_BLOCK; 
        sorted_ = matches;
        EASY_BLOCK("model_set", profiler::colors::Blue500);
        model_.set(matches, keys1, keys2);
        EASY_END_BLOCK; 

    }

    MODEL run()
    {
        EASY_FUNCTION(profiler::colors::Magenta);
        std::random_device rd;
        std::mt19937 gen(rd());
        EASY_BLOCK("iteration", profiler::colors::Blue500);
        for (int t = 1; t < iteration_; ++t)
        {
            std::cout << "n_ : " << n_ << std::endl;
            std::cout << "m_ : " << m_ << std::endl;
            std::cout << "Tpn_ :" << Tpn_ << std::endl;

            // 1. Choice of the hypothesis genration set
            if (t == static_cast<int>(Tpn_) && n_ < N_)
            {
                n_ += 1;
                double Tnn_ = (n_ + 1.0) / (n_ + 1.0 - m_) * Tn_;
                Tpn_ = Tpn_ + Tnn_ - Tn_;
                Tn_ = Tnn_;
            }

            // 2. Semi-random sample M of size m
            if (Tpn_ < t)
            {
                // if T'_{n} < t, then the sample contains m-1 points selected from U_{n-1} at random and u_{n}
                m_ -= 1;
            }
            // else select m poins from U_{n} at random

            // 3. Model parameter estimation
            EASY_BLOCK("iterate", profiler::colors::Blue500);
            if (iterate(gen))
            {
                if (best_model_inliers_ > threshold_)
                {
                    std::cout << "model!" << std::endl;
                    return model_;
                }
                if (model_.getInliers() > candidate_model_inliers_)
                {
                    candidate_ = model_;
                }
            }
            EASY_END_BLOCK; 
            // m값 복구
            m_ = 4;
        }
        std::cout << "candidate!" << std::endl;
        return candidate_;
    }
    bool iterate(std::mt19937 &gen)
    {
        EASY_FUNCTION(profiler::colors::Magenta);
        EASY_BLOCK("model run", profiler::colors::Blue500);
        model_.run(n_, gen);
        EASY_END_BLOCK;
        auto inliers = model_.getInliers();
        std::cout << "the number of inliers : " << inliers << std::endl;

        if (inliers > best_model_inliers_)
        {
            best_model_inliers_ = inliers;
            return true;
        }
        return false;
    }

    MODEL &getModel()
    {
        return model_;
    }

    auto maximality()
    {
        int n_inlier = static_cast<int>((1 - max_outlier_) * n_);

        double Pi{1.0};
        for (int i = 0; i < m_; ++i)
        {
            Pi *= (n_inlier - i) / (N_ - i);
        }

        return std::log10(eta_) / std::log10(1 - Pi);
    }

    // auto non_randomness()
    // {
    //     ()
    // }
};

#endif
