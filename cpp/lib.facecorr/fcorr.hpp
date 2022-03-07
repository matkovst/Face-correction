#pragma once

#include "math/image.hpp"
#include "math/skin_segm.hpp"

namespace fcorr
{

typedef struct
{
    int ksize = 3;
    float skin_thr = 0.25;
} Settings;

void identity(Image img, Image& out);

void correct(Image img, Image& out, Settings settings);

}