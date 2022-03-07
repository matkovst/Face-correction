#pragma once

#include "skin_segm.hpp"

float multigauss(const float x[3], const float mu[3], const float sigma[3], float w)
{
    float det = sigma[0] * sigma[1] * sigma[2];
    if (det == 0)
    {
        return 0.0f;
    }

    float e_coeff = 0;
    float mu_dev[3] = {x[0] - mu[0], x[1] - mu[1], x[2] - mu[2]};
    float tmp[3] = {mu_dev[0] * (1/sigma[0]), mu_dev[1] * (1/sigma[1]), mu_dev[2] * (1/sigma[2])};
    e_coeff = tmp[0] * mu_dev[0] + tmp[1] * mu_dev[1] + tmp[2] * mu_dev[2];
    e_coeff *= -0.5;

    float e = expf(e_coeff);

    float gauss = w * (e / sqrtf(powf(TWOPI, 3) * det));

    return gauss;
}


float skin_likelihood(const float pixel[3])
{
    float lhood = 0;
    for (int mode = 0; mode < 16; mode++)
    {
        const float _mean[3] = {Skin_Mus[mode][0], Skin_Mus[mode][1], Skin_Mus[mode][2]};
        const float _sigma[3] = {Skin_Sigmas[mode][0], Skin_Sigmas[mode][1], Skin_Sigmas[mode][2]};
        lhood += multigauss(pixel, _mean, _sigma, Skin_Ws[mode]);
    }
    return lhood;
}


float nonskin_likelihood(const float pixel[3])
{
    float lhood = 0;
    for (int mode = 0; mode < 16; mode++)
    {
        const float _mean[3] = {Nonskin_Mus[mode][0], Nonskin_Mus[mode][1], Nonskin_Mus[mode][2]};
        const float _sigma[3] = {Nonskin_Sigmas[mode][0], Nonskin_Sigmas[mode][1], Nonskin_Sigmas[mode][2]};
        lhood += multigauss(pixel, _mean, _sigma, Nonskin_Ws[mode]);
    }
    return lhood;
}


float segment_skin_pixel(const float pixel[3])
{
    float skin_prob = skin_likelihood(pixel) * SKIN_PRIOR;
    float nonskin_prob = nonskin_likelihood(pixel) * NONSKIN_PRIOR;
    float denom = skin_prob + nonskin_prob;
    if (denom == 0)
    {
        return 0.0f;
    }
    else
    {
        return (skin_prob / (denom));
    }
}