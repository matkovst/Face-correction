#include <cstring>

#include "image.hpp"

image make_empty_image(int w, int h, int c)
{
    image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

image make_image(int w, int h, int c)
{
    image out = make_empty_image(w, h, c);
    out.data = static_cast<float*>(calloc(h * w * c, sizeof(float)));
    return out;
}

image copy_image(image im)
{
    image copy = make_image(im.w, im.h, im.c);
    std::memcpy(copy.data, im.data, im.w * im.h * im.c * sizeof(float));
    return copy;
}

void free_image(image im)
{
    free(im.data);
}

image load_image(float* data, int w, int h, int c)
{
    if (!data)
    {
        fprintf(stderr, "Cannot load image\n");
        exit(0);
    }

    image im = make_image(w, h, c);
    for(int iC = 0; iC < c; ++iC)
    {
        for(int iH = 0; iH < h; ++iH)
        {
            for(int iW = 0; iW < w; ++iW)
            {
                const int index = iW + w * iH + w * h * iC; 
                im.data[index] = static_cast<float>(data[index]) / 255.0f;
            }
        }
    }

    return im;
}