#include "fcorr.hpp"

namespace fcorr
{

void identity(Image img, Image& out)
{
    out = copy_image(img);
}

void correct(Image img, Image& out, Settings settings)
{
//     /* Segment skin */
//     Image skinMask = make_image(img.w, img.h, 1);
// #ifdef WITH_OPENMP
//     #pragma omp parallel for num_threads(OMP_THREADS)
// #endif
//     for (int iH = 0; iH < img.h; ++iH)
//     {
//         for (int iW = 0; iW < img.w; ++iW)
//         {
//             const int b_index = get_index(iW, iH, 0, img.w, img.h);
//             const int g_index = get_index(iW, iH, 1, img.w, img.h);
//             const int r_index = get_index(iW, iH, 2, img.w, img.h);
//             const int out_index = b_index;
//             const float b_pixel = img.data[b_index];
//             const float g_pixel = img.data[g_index];
//             const float r_pixel = img.data[r_index];
//             const float pixel[3] = {b_pixel, g_pixel, r_pixel};
            
//             const float skin_prob = segment_skin_pixel(pixel);
//             skinMask.data[out_index] = (skin_prob >= settings.skin_thr) ? 255.0 : 0.0;
//         }
//     }
    
    /* Smooth img */
    Image boxFilter = make_box_filter(settings.ksize);
    Image boxedImg = convolve_image(img, boxFilter, 1);
    clamp_image(boxedImg);

    // bitwise_and(boxedImg, skinMask);

    out = from_image_to_cvmat(boxedImg);

    free_image(boxFilter);
    free_image(boxedImg);
}

}