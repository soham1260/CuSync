#include "../include/NvDecPipeline.h"
#include <cuda_runtime.h>

NvDecPipeline::NvDecPipeline() : demuxer(nullptr), decoder(nullptr), d_bgrFrame(nullptr),width(0), height(0), nv12Pitch(0), finished(false), pendingIdx(0) {}

NvDecPipeline::~NvDecPipeline() 
{
    if (d_bgrFrame) cudaFree(d_bgrFrame);
    delete decoder;
    delete demuxer;
}