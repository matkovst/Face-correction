#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
    
    /* Looping */
    clock_t tStart = clock();
    std::int64_t fno = 1;

    // TODO

    double elapsed = static_cast<double>(clock() - tStart) / CLOCKS_PER_SEC;
    double avgTime = elapsed / fno;
    std::cout << "Average time to process frame " << avgTime << " sec" << std::endl;

    std::cout << "Program successfully finished" << std::endl;
    return 0;
}