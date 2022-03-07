#pragma once

#include <stdio.h>
#include <stdlib.h>

typedef struct
{
    int w, h, c;
    float* data;
} image;

image make_empty_image(int w, int h, int c);

image make_image(int w, int h, int c);

image load_image(float* data, int w, int h, int c);

image copy_image(image im);

void free_image(image im);