#include <cuda.h>
#include <cuda_runtime.h>

#include "../include/NvDecPipeline.h"
#include "../Video_Codec_SDK_13.1.15/Samples/Utils/ColorSpace.h"

// Required by NVIDIA SDK Logger
simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

template <class COLOR24>
void Nv12ToColor24(uint8_t *dpNv12, int nNv12Pitch, uint8_t *dpBgr, int nBgrPitch, int nWidth, int nHeight, int iMatrix, bool video_full_range);

NvDecPipeline* createPipeline(const char* filePath, CUcontext cuContext) 
{
    NvDecPipeline* pipe = new NvDecPipeline();
    
    pipe->demuxer = new FFmpegDemuxer(filePath);
    pipe->decoder = new NvDecoder(
        cuContext, 
        true,   // Decoded frames stay in GPU
        FFmpeg2NvCodecId(pipe->demuxer->GetVideoCodec()),
        false,  // bLowLatency
        true    // Use pitched allocation for coalesced access
    );

    // Decode first few packets to initialize decoder/video dimensions
    uint8_t* videoPacket = nullptr;
    int videoBytes = 0;
    int64_t pts = 0;
    
    // Feed packets until decoder is initialized/emits at least one frame
    while (true) 
    {
        pipe->demuxer->DemuxVideo(&videoPacket, &videoBytes, &pts);
        if (videoBytes == 0) 
        {
            pipe->finished = true;
            break;
        }
        
        int frames = pipe->decoder->Decode(videoPacket, videoBytes, 0, pts);
        if (frames > 0) 
        {
            for (int i = 0; i < frames; i++) 
            {
                pipe->pendingFrames.push_back(pipe->decoder->GetFrame());
            }
            pipe->pendingIdx = 0;
            break;
        }
    }

    pipe->width = pipe->decoder->GetWidth();
    pipe->height = pipe->decoder->GetHeight();
    pipe->nv12Pitch = pipe->decoder->GetDeviceFramePitch();

    // Allocate BGR output buffer on GPU
    int bgrSize = pipe->width * pipe->height * 3;
    cudaMalloc(&pipe->d_bgrFrame, bgrSize);

    return pipe;
}

uint8_t* getNextBgrFrame(NvDecPipeline* pipe) 
{
    if (pipe->finished) return nullptr;

    uint8_t* pNv12Frame = nullptr;

    // If there are already frames in pendingFrames buffer, use them
    if (pipe->pendingIdx < (int)pipe->pendingFrames.size()) 
    {
        pNv12Frame = pipe->pendingFrames[pipe->pendingIdx++];
        
        // If all pending frames are onsumed, clear the list
        if (pipe->pendingIdx >= (int)pipe->pendingFrames.size()) 
        {
            pipe->pendingFrames.clear();
            pipe->pendingIdx = 0;
        }
    } 
    else 
    {
        uint8_t* pVideo = nullptr;
        int nVideoBytes = 0;
        int64_t pts = 0;

        // Feed packets until decoder is initialized/emits at least one frame
        while (true) 
        {
            pipe->demuxer->DemuxVideo(&pVideo, &nVideoBytes, &pts);

            int nFrames = pipe->decoder->Decode(pVideo, nVideoBytes, 0, pts);
            
            if (nVideoBytes == 0 && nFrames == 0) 
            {
                pipe->finished = true;
                return nullptr;
            }

            if (nFrames > 0) 
            {
                // First frame is to be returned, rest go to pendingFrames buffer
                pNv12Frame = pipe->decoder->GetFrame();
                for (int i = 1; i < nFrames; i++) 
                {
                    pipe->pendingFrames.push_back(pipe->decoder->GetFrame());
                }
                pipe->pendingIdx = 0;
                break;
            }
        }
    }

    if (!pNv12Frame) return nullptr;

    // NV12 -> BGR24
    Nv12ToColor24<BGR24>(pNv12Frame,pipe->nv12Pitch,pipe->d_bgrFrame,pipe->width*3,pipe->width, pipe->height,0,false);

    return pipe->d_bgrFrame;
}