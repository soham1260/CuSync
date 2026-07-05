#include "../include/NvDecPipeline.h"

NvDecPipeline::NvDecPipeline() : demuxer(nullptr), decoder(nullptr), d_bgrFrame(nullptr) {}

NvDecPipeline::~NvDecPipeline() 
{
    if (d_bgrFrame) cudaFree(d_bgrFrame);
    delete decoder;
    delete demuxer;
}