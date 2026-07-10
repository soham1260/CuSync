#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#include "VideoProcessor.h"
#include "GaussianBlurFilter.h"
#include "ChromaKeyFilter.h"
#include "GrayscaleFilter.h"
#include "SepiaFilter.h"
#include "VignetteFilter.h"
#include "BloomFilter.h"
#include "PixelationFilter.h"
#include "MotionBlurFilter.h"
#include "FisheyeFilter.h"
#include "../include/NvDecPipeline.h"
#include "../include/NvVideoProcessor.h"

void onBlurTrackbar(int pos, void* userdata) 
{
    if (userdata) 
    {
        GaussianBlurFilter* blurFilter = static_cast<GaussianBlurFilter*>(userdata);
        float sigma = (float)pos / 10.0f;
        if (sigma < 0.1f) sigma = 0.1f;
        blurFilter->updateParameters({{"gaussian_sigma", sigma}});
    }
}

int main(int argc, char** argv) 
{
    if (argc < 2) 
    {
        std::cerr << "Usage: " << argv[0] << " <fg_video_path> [bg_video_path]" << std::endl;
        return -1;
    }

    // Explicit cuda context needed by decoder
    CUcontext cuContext = NULL;
    cuInit(0);
    CUdevice cuDevice = 0;
    CUctxCreateParams params = { NULL, NULL, 0 };
    cuCtxCreate(&cuContext,&params, 0, cuDevice);

    // Create pipelines for fetching frames
    NvDecPipeline* fgPipe = createPipeline(argv[1], cuContext);
    NvDecPipeline* bgPipe = NULL;
    if (argc >= 3) 
    {
        bgPipe = createPipeline(argv[2], cuContext);
    }

    // Get source FPS from demuxer instead of OpenCV
    AVStream* videoStream = fgPipe->demuxer->GetVideoStream();
    double source_fps = 30.0;
    if (videoStream && videoStream->avg_frame_rate.den > 0)
    {
        source_fps = av_q2d(videoStream->avg_frame_rate);
    }
    if (source_fps <= 0) source_fps = 30.0;
    auto frame_duration = std::chrono::microseconds((long long)(1000000.0 / source_fps));

    NvVideoProcessor processor(fgPipe->width, fgPipe->height, 3);
    int w = fgPipe->width;
    int h = fgPipe->height;
    int c = 3;
    
    GaussianBlurFilter* blurFilter = new GaussianBlurFilter(w, h, c, 3.0f, true);
    processor.addFilter(blurFilter);
    processor.addFilter(new ChromaKeyFilter(120.0f, 30.0f, 80.0f, 0.3f, 0.3f));

    cv::namedWindow("Video Frame", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Blur Sigma", "Video Frame", nullptr, 100, onBlurTrackbar, blurFilter);
    cv::setTrackbarPos("Blur Sigma", "Video Frame", 30);
    
    // processor.addFilter(new GrayscaleFilter());
    // processor.addFilter(new GrayscaleFilter());
    // processor.addFilter(new SepiaFilter());
    // processor.addFilter(new VignetteFilter(1.0f, 1.0f));
    // processor.addFilter(new BloomFilter(w, h, c, 180.0f, 5.0f));
    // processor.addFilter(new PixelationFilter(15));
    // processor.addFilter(new MotionBlurFilter(w, h, c, 0.8f));
    // processor.addFilter(new FisheyeFilter(w, h, c, 0.5f));

    // Frame 0
    uint8_t* d_fg_frame = getNextBgrFrame(fgPipe);
    uint8_t* d_bg_frame = bgPipe ? getNextBgrFrame(bgPipe) : NULL;
    if (d_fg_frame) processor.processFrameAsync(d_fg_frame, d_bg_frame, 0);

    // Frame 1
    d_fg_frame = getNextBgrFrame(fgPipe);
    d_bg_frame = bgPipe ? getNextBgrFrame(bgPipe) : NULL;
    if (d_fg_frame) processor.processFrameAsync(d_fg_frame, d_bg_frame, 1);

    int displayStream = 0;
    int queueStream = 2;

    auto s = std::chrono::high_resolution_clock::now();
    
    auto next_frame_target = std::chrono::steady_clock::now() + frame_duration;
    bool isPaused = false;

    while (1) 
    {
        // Read next frame only if not paused
        if (!isPaused) 
        {
            d_fg_frame = getNextBgrFrame(fgPipe);
            if (!d_fg_frame) break;

            d_bg_frame = bgPipe ? getNextBgrFrame(bgPipe) : NULL;
        }

        processor.processFrameAsync(d_fg_frame, d_bg_frame, queueStream); // frame is already in GPU

        unsigned char* processed_data = processor.syncAndGetFrame(displayStream);

        cv::Mat output(h, w, CV_8UC3, processed_data);
        cv::imshow("Video Frame", output);

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') break;
        if (key == 'p' || key == ' ') 
            isPaused = !isPaused;

        if (!isPaused) 
        {
            std::this_thread::sleep_until(next_frame_target);
            next_frame_target += frame_duration;
        } 
        else 
        {
            next_frame_target = std::chrono::steady_clock::now() + frame_duration;
        }

        displayStream = (displayStream + 1) % 3;
        queueStream = (queueStream + 1) % 3; 
    }
    
    auto e = std::chrono::high_resolution_clock::now();  
    std::chrono::duration<double> total_seconds = e - s;
    std::cout << "Total Playback Time: " << total_seconds.count() << " s" << std::endl;

    for(int i = 0; i < 2; i++) 
    {
        unsigned char* processed_data = processor.syncAndGetFrame(displayStream);
        cv::Mat output(h, w, CV_8UC3, processed_data);
        cv::imshow("Video Frame", output);
        cv::waitKey(1);
        displayStream = (displayStream + 1) % 3;
    }

    delete fgPipe;
    delete bgPipe; 
    cv::destroyAllWindows();
    return 0;
}