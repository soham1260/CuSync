cmake_minimum_required(VERSION 3.30)

project(VideoProcessorNvdec LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(OpenCV_DIR "C:/opencv/opencv/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ffmpeg and Nvidia Video Codec SDK root
set(FFMPEG_ROOT "${CMAKE_SOURCE_DIR}/ffmpeg-master-latest-win64-gpl-shared/ffmpeg-master-latest-win64-gpl-shared")
# set(FFMPEG_ROOT "${CMAKE_SOURCE_DIR}/ffmpeg-master-latest-win64-gpl/ffmpeg-master-latest-win64-gpl")
set(NVCODEC_ROOT "${CMAKE_SOURCE_DIR}/Video_Codec_SDK_13.1.15")

# ffmpeg includes and libraries
set(FFMPEG_INCLUDE_DIR "${FFMPEG_ROOT}/include")
set(FFMPEG_LIB_DIR "${FFMPEG_ROOT}/lib")

# NVIDIA Video Codec SDK includes and libraries
set(NVCODEC_INCLUDE_DIR "${NVCODEC_ROOT}/Samples/NvCodec")
set(NVCODEC_LIB_DIR "${NVCODEC_ROOT}/Lib/win/x64")

add_executable(VideoProcessorNvdec

    src/main.cpp
    src/NvVideoProcessor.cu
    src/NvDecPipeline.cpp
    src/NvDecPipelineHelpers.cpp

    src/filters/GaussianBlurFilter.cu
    src/filters/ChromaKeyFilter.cu
    src/filters/GrayscaleFilter.cu
    src/filters/SepiaFilter.cu
    src/filters/VignetteFilter.cu
    src/filters/BloomFilter.cu
    src/filters/PixelationFilter.cu
    src/filters/MotionBlurFilter.cu
    src/filters/FisheyeFilter.cu
    
    ${NVCODEC_ROOT}/Samples/NvCodec/NvDecoder/NvDecoder.cpp
    ${NVCODEC_ROOT}/Samples/Utils/ColorSpace.cu
)

target_include_directories(VideoProcessorNvdec PRIVATE
    include
    include/filters
    ${FFMPEG_INCLUDE_DIR} # libavformat/avformat.h required by demuxer
    ${NVCODEC_INCLUDE_DIR} # NvDecoder/NvDecoder.h
    # ${NVCODEC_UTILS_DIR} # Already provided relative path 
)

target_compile_options(VideoProcessorNvdec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=native>)

target_link_directories(VideoProcessorNvdec PRIVATE ${FFMPEG_LIB_DIR} ${NVCODEC_LIB_DIR})

target_link_libraries(VideoProcessorNvdec
    ${OpenCV_LIBS}
    
    # ffmpeg
    avformat
    avcodec
    avutil
    
    # NVDEC
    nvcuvid
    
    CUDA::cuda_driver
)

# using ffmpeg shared version, copy dlls
file(GLOB FFMPEG_DLLS "${FFMPEG_ROOT}/bin/*.dll")

add_custom_command(TARGET VideoProcessorNvdec POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${FFMPEG_DLLS}
        $<TARGET_FILE_DIR:VideoProcessorNvdec>
    COMMENT "Copying ffmpef DLLs to output directory"
)
