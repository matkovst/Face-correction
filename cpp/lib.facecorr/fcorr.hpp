#pragma once

#include "math/image.hpp"
#include "math/skin_segm.hpp"

namespace fcorr
{

typedef struct
{
    float h = 0;
    float s = 0;
    float v = 0;
} HSVRange;

void identity(Image img, Image& out); // test function

class FaceCorrector final
{
public:

    typedef struct
    {
        int ksize = 3;
        float skin_thr = 0.25;
    } Settings;

    FaceCorrector();

    ~FaceCorrector();

    void correct(Image img, Image& out, Settings settings);

    const Image& skinMask() const noexcept;

private:
    bool m_firstFrame { true };
    Image m_skinMask;
    HSVRange m_skinLBound;
    HSVRange m_skinRBound;
};

}