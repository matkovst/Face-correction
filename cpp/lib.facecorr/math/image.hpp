#pragma once

#include <stdio.h>
#include <stdlib.h>

namespace fcorr
{

typedef struct
{
    int w, h, c;
    float* data;
} Image;

/* Memory management */

Image make_empty_image(int w, int h, int c);

Image make_image(int w, int h, int c);

Image load_image_from_cvmat(float* data, int w, int h, int c, float scale = 0.00392156862745098f);

Image from_image_to_cvmat(Image im);

Image copy_image(Image im);

void free_image(Image im);


/* Basics */

int get_index(int x, int y, int z, int w, int h);

int get_cvmat_index(int x, int y, int z, int w, int h, int c);

float get_pixel(Image im, int x, int y, int c);

void set_pixel(Image im, int x, int y, int c, float v);

void clamp_image(Image im);

float three_way_max(float a, float b, float c);

float three_way_min(float a, float b, float c);


/* Filtering & Processing */

Image convolve_image(Image im, Image filter, int preserve);

Image make_box_filter(int w);

void rgb_to_hsv(Image im);

void hsv_to_rgb(Image im);

/* bitwise_and in-place */
void hsv_inRange_inplace(Image im, int min_h, int min_s, int min_v, int max_h, int max_s, int max_v);

void hsv_inRange(Image im, Image& out, int min_h, int min_s, int min_v, int max_h, int max_s, int max_v);

void bitwise_and(Image im, Image mask);

}