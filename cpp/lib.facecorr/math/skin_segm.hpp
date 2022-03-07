#pragma once

#include <stdio.h>
#include <iostream>
#include <cmath>

#include "skin_weights.hpp"

namespace fcorr
{

float multigauss(const float x[3], const float mu[3], const float sigma[3], float w);

float skin_likelihood(const float pixel[3]);

float nonskin_likelihood(const float pixel[3]);

float segment_skin_pixel(const float pixel[3]);

}