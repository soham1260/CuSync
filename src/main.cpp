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

    cv::VideoCapture fg(argv[1]);
    if (!fg.isOpened()) 
    {
        std::cerr << "Error: Could not open foreground video: " << argv[1] << std::endl;
        return -1;
    }

    bool has_bg = false;
    cv::VideoCapture bg;
    if (argc >= 3) 
    {
        bg.open(argv[2]);
        if (bg.isOpened()) 
        {
            has_bg = true;
        }
        else 
        {
            std::cout << "Warning: Could not open background video: " << argv[2] << ". Using black background instead." << std::endl;
        }
    }

    double source_fps = fg.get(cv::CAP_PROP_FPS);
    if (source_fps <= 0) source_fps = 30.0;
    auto frame_duration = std::chrono::microseconds((long long)(1000000.0 / source_fps));

    cv::Mat fg_frame, bg_frame;
    fg >> fg_frame;
    if (fg_frame.empty()) 
    {
        std::cerr << "Error: Foreground video has no frames." << std::endl;
        return -1;
    }

    if (has_bg) 
    {
        bg >> bg_frame;
        if (bg_frame.empty()) 
        {
            has_bg = false;
        }
    } 

    // reset back to 0
    fg.set(cv::CAP_PROP_POS_FRAMES, 0);
    if (has_bg) 
    {
        bg.set(cv::CAP_PROP_POS_FRAMES, 0);
    }

    int w = fg_frame.cols;
    int h = fg_frame.rows;
    int c = fg_frame.channels();

    VideoProcessor processor(w, h, c);
    
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
    fg >> fg_frame;
    if (has_bg) 
    {
        bg >> bg_frame;
        if (bg_frame.empty()) 
        {
            bg.set(cv::CAP_PROP_POS_FRAMES, 0);
            bg >> bg_frame;
        }
        if (fg_frame.size() != bg_frame.size()) 
            cv::resize(bg_frame, bg_frame, fg_frame.size());
    } 
    processor.processFrameAsync(fg_frame.data, bg_frame.data, 0);

    // Frame 1
    fg >> fg_frame;
    if (has_bg) 
    {
        bg >> bg_frame;
        if (bg_frame.empty()) 
        {
            bg.set(cv::CAP_PROP_POS_FRAMES, 0);
            bg >> bg_frame;
        }
        if (fg_frame.size() != bg_frame.size()) 
            cv::resize(bg_frame, bg_frame, fg_frame.size());
    } 
    processor.processFrameAsync(fg_frame.data, bg_frame.data, 1);

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
            fg >> fg_frame; 
            if (fg_frame.empty()) break;
            
            if (has_bg) 
            {
                bg >> bg_frame;
                if (bg_frame.empty()) 
                {
                    bg.set(cv::CAP_PROP_POS_FRAMES, 0);
                    bg >> bg_frame; 
                }
                if (fg_frame.size() != bg_frame.size()) 
                    cv::resize(bg_frame, bg_frame, fg_frame.size());
            } 
        }


        processor.processFrameAsync(fg_frame.data, bg_frame.data, queueStream);

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

    fg.release(); 
    bg.release(); 
    cv::destroyAllWindows();
    return 0;
}