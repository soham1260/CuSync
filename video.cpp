#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

#include "VideoProcessor.h"
#include "GaussianBlurFilter.h"
#include "ChromaKeyFilter.h"
#include "GrayscaleFilter.h"
#include "SepiaFilter.h"
#include "VignetteFilter.h"

int main() 
{
    cv::VideoCapture fg("../../fg.mp4");
    cv::VideoCapture bg("../../bg.mp4");
    
    if (!fg.isOpened() || !bg.isOpened()) return -1;

    double source_fps = fg.get(cv::CAP_PROP_FPS);
    if (source_fps <= 0) source_fps = 30.0;
    auto frame_duration = std::chrono::microseconds((long long)(1000000.0 / source_fps));

    cv::Mat fg_frame, bg_frame;
    fg >> fg_frame; bg >> bg_frame;
    fg.set(cv::CAP_PROP_POS_FRAMES, 0); bg.set(cv::CAP_PROP_POS_FRAMES, 0);

    int w = fg_frame.cols;
    int h = fg_frame.rows;
    int c = fg_frame.channels();

    VideoProcessor processor(w, h, c);
    
    processor.addFilter(new GaussianBlurFilter(w, h, c, 3.0f, true));
    processor.addFilter(new ChromaKeyFilter(120.0f, 30.0f, 80.0f, 0.3f, 0.3f));
    
    // processor.addFilter(new GrayscaleFilter());
    // processor.addFilter(new SepiaFilter());
    processor.addFilter(new VignetteFilter(1.0f, 1.0f));

    fg >> fg_frame; bg >> bg_frame;
    if(fg_frame.size() != bg_frame.size()) 
        cv::resize(bg_frame, bg_frame, fg_frame.size());
    processor.processFrameAsync(fg_frame.data, bg_frame.data, 0);

    fg >> fg_frame; bg >> bg_frame;
    if(fg_frame.size() != bg_frame.size()) 
        cv::resize(bg_frame, bg_frame, fg_frame.size());
    processor.processFrameAsync(fg_frame.data, bg_frame.data, 1);

    int displayStream = 0;
    int queueStream = 2;

    auto s = std::chrono::high_resolution_clock::now();
    
    auto next_frame_target = std::chrono::steady_clock::now() + frame_duration;

    while (1) 
    {
        fg >> fg_frame; 
        bg >> bg_frame;
        
        if (fg_frame.empty()) break;
        if (bg_frame.empty()) 
        {
            bg.set(cv::CAP_PROP_POS_FRAMES, 0);
            bg >> bg_frame; 
        }
        if (fg_frame.size() != bg_frame.size()) 
            cv::resize(bg_frame, bg_frame, fg_frame.size());

        processor.processFrameAsync(fg_frame.data, bg_frame.data, queueStream);

        unsigned char* processed_data = processor.syncAndGetFrame(displayStream);

        cv::Mat output(h, w, CV_8UC3, processed_data);
        cv::imshow("Video Frame", output);

        if (cv::waitKey(1) >= 0) break;

        std::this_thread::sleep_until(next_frame_target);

        next_frame_target += frame_duration;

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

    fg.release(); 
    bg.release(); 
    cv::destroyAllWindows();
    return 0;
}