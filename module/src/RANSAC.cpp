#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <tuple>

#include "RANSAC.hpp"

/*
    N: Total number of samples.

    m: Complexity of the geometric model.

    t: Number of iterations of the loop.

    n*: Termination length (number of samples to consider before stopping).

    TN: Parameter that defines after how many samples PROSAC becomes equivalent to RANSAC.

    Tn: Average number of samples from the n top-ranked points.

    Tn': Ceiled value of Tn
*/


