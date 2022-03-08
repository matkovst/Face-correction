#include <iostream>
#include <cstring>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "lib.facecorr/fcorr.hpp"

int main(int argc, char *argv[])
{
    std::cout << "Program started. Have fun!" << std::endl;

    /* Parse cmd args */
    if (argc != 2) 
    {
        std::cerr << "usage: %s input_file\n"  
            << "Demo program to show how to use face correction library.\n", argv[0];
        return 1;
    }

    const std::string inputpath = argv[1];
    cv::VideoCapture capture;
    if (inputpath == "0" || inputpath == "/dev/video0")
        capture.open(0);
    else
        capture.open(inputpath);
    if( !capture.isOpened() )
    {
        std::cout << "Could not initialize capturing..." << std::endl;
        return 0;
    }

    cv::Mat frame;
    capture >> frame;
    if( frame.empty() )
    {
        std::cout << "Could not capture frame..." << std::endl;
        return 0;
    }
    
    /* Looping */
    clock_t tStart = clock();
    std::int64_t fno = 1;

    fcorr::FaceCorrector fcorrector;
    fcorr::FaceCorrector::Settings settings;
    settings.ksize = 7;
    settings.skin_thr = 0.25;

    while (true)
    {
        capture >> frame;
        if( frame.empty())
            break;
        ++fno;
        
        // make input
        cv::Mat frame32f;
        frame.convertTo(frame32f, CV_32FC3);
        // std::vector<float> frame32fvec;
        // frame32fvec.assign((float*)frame32f.data, (float*)frame32f.data + frame32f.total() * frame32f.channels());
        fcorr::Image im = fcorr::load_image_from_cvmat((float*)frame32f.data /* frame32fvec.data() */,
                                            frame32f.cols,
                                            frame32f.rows,
                                            frame32f.channels());
        fcorr::Image imOut;

        // magic
        {
            // fcorr::identity(im, imOut);
            fcorrector.correct(im, imOut, settings);
        }

        // make output
        cv::Mat out(imOut.h, imOut.w, CV_32FC3, (void*)imOut.data);

        // fcorr::Image skinMask = fcorr::from_image_to_cvmat(fcorrector.skinMask());
        // cv::Mat outSkinMask(skinMask.h, skinMask.w, CV_32FC1, (void*)skinMask.data);
        // std::cout << cv::sum(outSkinMask) << std::endl;
        // cv::imshow("outSkinMask", outSkinMask);
        // fcorr::free_image(skinMask);

        cv::imshow("Demo", out);

        fcorr::free_image(im);
        fcorr::free_image(imOut);

        const char c = static_cast<char>(cv::waitKey(10));
        if(27 == c)
            break;
    }
    capture.release();
    cv::destroyAllWindows();

    double elapsed = static_cast<double>(clock() - tStart) / CLOCKS_PER_SEC;
    double avgTime = elapsed / fno;
    std::cout << "Average time to process frame " << avgTime << " sec" << std::endl;

    std::cout << "Program successfully finished" << std::endl;
    return 0;
}