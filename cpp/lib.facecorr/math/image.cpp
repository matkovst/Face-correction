#include <cstring>
#include <cassert>

#include "image.hpp"

namespace fcorr
{

int get_index(int x, int y, int z, int w, int h)
{
    return x + w * y + w * h * z;
}

int get_cvmat_index(int x, int y, int z, int w, int h, int c)
{
    return z + c * x + c * w * y;
}

float get_pixel(Image im, int x, int y, int c)
{
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (c < 0) c = 0;
    if (x > im.w - 1) x = im.w - 1;
    if (y > im.h - 1) y = im.h - 1;
    if (c > im.c - 1) c = im.c - 1;
    float pixel = im.data[im.w * im.h * c + im.w * y + x];
    return pixel;
}

void set_pixel(Image im, int x, int y, int c, float v)
{
    if (x < 0 || x > im.w) return;
    if (y < 0 || y > im.h) return;
    im.data[im.w * im.h * c + im.w * y + x] = v;
}

void clamp_image(Image im)
{
    float pixel;

#ifdef WITH_OPENMP
    #pragma omp parallel for num_threads(OMP_THREADS)
#endif
    for (int c = 0; c < im.c; ++c)
    {
        for (int w = 0; w < im.w; ++w)
        {
            for (int h = 0; h < im.h; ++h)
            {
                pixel = get_pixel(im, w, h, c);
                if (pixel < 0) set_pixel(im, w, h, c, 0);
                if (pixel > 1) set_pixel(im, w, h, c, 1);
            }
        }
    }
}

Image make_empty_image(int w, int h, int c)
{
    Image out;
    out.data = 0;
    out.h = h;
    out.w = w;
    out.c = c;
    return out;
}

Image make_image(int w, int h, int c)
{
    Image out = make_empty_image(w, h, c);
    out.data = static_cast<float*>(calloc(h * w * c, sizeof(float)));
    return out;
}

Image copy_image(Image im)
{
    Image copy = make_image(im.w, im.h, im.c);
    std::memcpy(copy.data, im.data, im.w * im.h * im.c * sizeof(float));
    return copy;
}

void free_image(Image im)
{
    free(im.data);
}

Image load_image_from_cvmat(float* data, int w, int h, int c, float scale)
{
    if (!data)
    {
        fprintf(stderr, "[load_image_from_cvmat] Cannot load image\n");
        exit(0);
    }

    Image im = make_image(w, h, c);
#ifdef WITH_OPENMP
    #pragma omp parallel for num_threads(OMP_THREADS)
#endif
    for (int iC = 0; iC < c; ++iC)
    {
        for (int iH = 0; iH < h; ++iH)
        {
            for (int iW = 0; iW < w; ++iW)
            {
                const int dst_index = get_index(iW, iH, iC, w, h);
                const int src_index = get_cvmat_index(iW, iH, iC, w, h, c);
                im.data[dst_index] = static_cast<float>(data[src_index]) * scale;
            }
        }
    }

    return im;
}

Image from_image_to_cvmat(Image im)
{
    if (!im.data)
    {
        fprintf(stderr, "[from_image_to_cvmat] Cannot load image\n");
        exit(0);
    }

    Image out = make_image(im.w, im.h, im.c);
#ifdef WITH_OPENMP
    #pragma omp parallel for num_threads(OMP_THREADS)
#endif
    for (int iC = 0; iC < im.c; ++iC)
    {
        for (int iH = 0; iH < im.h; ++iH)
        {
            for (int iW = 0; iW < im.w; ++iW)
            {
                const int src_index = get_index(iW, iH, iC, im.w, im.h);
                const int dst_index = get_cvmat_index(iW, iH, iC, im.w, im.h, im.c);
                out.data[dst_index] = static_cast<float>(im.data[src_index]);
            }
        }
    }

    return out;
}

Image convolve_image(Image im, Image filter, int preserve)
{
    assert(im.c == filter.c || filter.c == 1);

    Image conv_im;
    if (preserve == 1)
        conv_im = make_image(im.w, im.h, im.c);
    else
        conv_im = make_image(im.w, im.h, 1);

#ifdef WITH_OPENMP
    #pragma omp parallel for num_threads(OMP_THREADS)
#endif
    for (int ch = 0; ch < im.c; ++ch)
    {
        for (int y = 0; y < im.h; ++y)
        {
            for (int x = 0; x < im.w; ++x)
            {
                float sum = 0;
                for (int fy = 0; fy < filter.h; ++fy)
                {
                    for (int fx = 0; fx < filter.w; ++fx)
                    {
                        int fc = 0;
                        if (im.c == filter.c)
                        {
                            fc = ch;
                        }
                        sum += filter.data[fc * filter.h * filter.w + fy * filter.w + fx] *
                            get_pixel(im, x + fx - (filter.w / 2), y + fy - (filter.h / 2), ch);
                    }
                }
                if (preserve == 1)
                    conv_im.data[ch * conv_im.h * conv_im.w + y * conv_im.w + x] = sum;
                else
                    conv_im.data[y * conv_im.w + x] += sum;
            }
        }
    }

    return conv_im;
}

Image make_box_filter(int w)
{
	Image box_filter = make_image(w, w, 1);
	const float size = static_cast<float>(w * w);
	for (int i = 0; i < w; ++i)
    {
		for (int j = 0; j < w; ++j)
        {
			set_pixel(box_filter, j, i, 0, (1.0f / size));
		}
    }
    return box_filter;
}

void bitwise_and(Image im, Image mask)
{
#ifdef WITH_OPENMP
    #pragma omp parallel for num_threads(OMP_THREADS)
#endif
    for (int iC = 0; iC < im.c; ++iC)
    {
        for (int iH = 0; iH < im.h; ++iH)
        {
            for (int iW = 0; iW < im.w; ++iW)
            {
                const int index = get_index(iW, iH, iC, im.w, im.h);
                const int mask_index = get_index(iW, iH, 0, im.w, im.h);
                if (0.0f == mask.data[mask_index])
                    im.data[index] = 0.0f;
            }
        }
    }
}

}