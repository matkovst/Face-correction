#include "fcorr.hpp"

namespace fcorr
{

void identity(Image img, Image& out)
{
    out = copy_image(img);
}

FaceCorrector::FaceCorrector()
{
    m_skinLBound.h = 0.0f / 255.0f;
    m_skinLBound.s = 15.0f / 255.0f;
    m_skinLBound.v = 0.0f / 255.0f;

    m_skinRBound.h = 17.0f / 255.0f;
    m_skinRBound.s = 170.0f / 255.0f;
    m_skinRBound.v = 255.0f / 255.0f;
}

FaceCorrector::~FaceCorrector()
{
    free_image(m_skinMask);
}

void FaceCorrector::correct(Image img, Image& out, Settings settings)
{
    if (m_firstFrame)
        m_skinMask = make_image(img.w, img.h, 1);

    /* Smooth img */
    Image boxFilter = make_box_filter(settings.ksize);
    Image boxedImg = convolve_image(img, boxFilter, 1);
    clamp_image(boxedImg);

    /* "Enlight" skin pixels */

    // rgb_to_hsv(img);
    // hsv_inRange(img, m_skinMask, m_skinLBound.h, m_skinLBound.s, m_skinLBound.v, 
    //                                 m_skinRBound.h, m_skinRBound.s, m_skinRBound.v);
    // hsv_to_rgb(img);

#ifdef WITH_OPENMP
    #pragma omp parallel for num_threads(OMP_THREADS)
#endif
    for (int iH = 0; iH < img.h; ++iH)
    {
        for (int iW = 0; iW < img.w; ++iW)
        {
            const int b_index = get_index(iW, iH, 0, img.w, img.h);
            const int g_index = get_index(iW, iH, 1, img.w, img.h);
            const int r_index = get_index(iW, iH, 2, img.w, img.h);
            const int mask_index = b_index;
            const float b_pixel = img.data[b_index];
            const float g_pixel = img.data[g_index];
            const float r_pixel = img.data[r_index];

            if (r_pixel <= (97.0f / 255.0f) || g_pixel <= (40.0f / 255.0f) || b_pixel <= (20.0f / 255.0f))
            {
                m_skinMask.data[mask_index] = 0.0f;
                break;
            }
            if ((r_pixel - g_pixel) <= (15.0f / 255.0f))
            {
                m_skinMask.data[mask_index] = 0.0f;
                break;
            }
            if (!(r_pixel > g_pixel && g_pixel > b_pixel))
            {
                m_skinMask.data[mask_index] = 0.0f;
                break;
            }
            // if (!((three_way_max(r_pixel, g_pixel, b_pixel) - three_way_min(r_pixel, g_pixel, b_pixel)) > (15.0f / 255.0f)))
            // {
            //     m_skinMask.data[mask_index] = 0.0f;
            //     break;
            // }

            m_skinMask.data[mask_index] = 1.0f;
        }
    }

//     /* Apply smoothing to skin */
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
//             const int mask_index = get_index(iW, iH, 0, img.w, img.h);

//             if (0.0f == m_skinMask.data[mask_index])
//             {
//                 boxedImg.data[b_index] = img.data[b_index];
//                 boxedImg.data[g_index] = img.data[g_index];
//                 boxedImg.data[r_index] = img.data[r_index];
//             }
//         }
//     }

    out = from_image_to_cvmat(boxedImg);

    free_image(boxFilter);
    free_image(boxedImg);
    m_firstFrame = false;
}

const Image& FaceCorrector::skinMask() const noexcept
{
    return m_skinMask;
}

}