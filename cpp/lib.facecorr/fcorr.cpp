#include "fcorr.hpp"

void identity(const image& img, image& out)
{
    out = copy_image(img); // identity
}