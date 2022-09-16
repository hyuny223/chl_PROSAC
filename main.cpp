#include <iostream>
#include <algorithm>

#include "eigen3/Eigen/Dense"

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

    std::string path(argv[1]);

    DetectMatch dm(path);
    auto [matches, keys1, keys2] = dm.run();

    Homography h;
    PROSAC<Homography> prosac(matches, keys1, keys2, 100, 100, h);
    prosac.run();
    auto model = prosac.getModel();
    dm.show(model.getHomography());

}