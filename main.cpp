#include <iostream>
#include <algorithm>

#include "eigen3/Eigen/Dense"
#include "easy/profiler.h"

#include "detect.hpp"
#include "homography.hpp"
#include "RANSAC.hpp"

int main(int argc, char** argv)
{
    if(argc > 2)
    {
        std::cerr << "just put the path" << std::endl;
        return -1;
    }
    EASY_PROFILER_ENABLE;

    std::string path(argv[1]);

    DetectMatch dm(path);
    auto [matches, keys1, keys2] = dm.run();
    auto s = std::chrono::steady_clock::now();
    Homography h;
    PROSAC<Homography> prosac(matches, keys1, keys2, 100, 1, h);
    prosac.run();
    auto e = std::chrono::steady_clock::now();
    std::chrono::duration<double> t = e - s;
    std::cout << std::endl;
    std::cout << "my time : " << t.count() << std::endl;

    auto model = prosac.getModel();
    std::cout << "my error : " << model.error_ << std::endl;
    std::cout << std::endl;
    dm.show(model.getHomography());
    // profiler::dumpBlocksToFile("../profiler/dev_profile.prof");
}
