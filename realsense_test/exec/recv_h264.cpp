#include "RealsenseMsg.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include "wlrobot/robot/channel/channel_subscriber.hpp"

#define LOG_USE_COLOR 1
#define LOG_USE_PREFIX 1
#include "utility/logger.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace custom = wlrobot::robot::channel;

static AVCodecContext* dec_ctx = nullptr;
static SwsContext* sws_ctx = nullptr;
static AVFrame* frame = nullptr;

void init_decoder(int width, int height) {
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) {
        FRC_ERROR("H.264 decoder not found");
        exit(1);
    }
    dec_ctx = avcodec_alloc_context3(codec);
    if (avcodec_open2(dec_ctx, codec, nullptr) < 0) {
        FRC_ERROR("Could not open decoder");
        exit(1);
    }
    frame = av_frame_alloc();
    sws_ctx = sws_getContext(
        width, height, AV_PIX_FMT_YUV420P,
        width, height, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, nullptr, nullptr, nullptr);
}

void OnRealsenseMsg(const RealsenseMsg& msg) {
    if (!dec_ctx) {
        init_decoder(msg.width, msg.height);
    }

    AVPacket* pkt = av_packet_alloc();
    pkt->data = msg.color_frame._buffer;
    pkt->size = msg.color_frame._length;

    if (avcodec_send_packet(dec_ctx, pkt) < 0) {
        FRC_ERROR("send_packet failed");
        av_packet_free(&pkt);
        return;
    }

    while (avcodec_receive_frame(dec_ctx, frame) == 0) {
        cv::Mat img(msg.height, msg.width, CV_8UC3);
        uint8_t* dest[4] = { img.data, nullptr, nullptr, nullptr };
        int dest_linesize[4] = { static_cast<int>(msg.width) * 3, 0, 0, 0 };

        sws_scale(sws_ctx,
                  frame->data, frame->linesize,
                  0, msg.height,
                  dest, dest_linesize);

        cv::imshow("Viewer H264", img);
        cv::waitKey(1);
    }

    av_packet_free(&pkt);
}

int main(int argc, char** argv) {
    FRC_INFO("Starting Realsense H.264 subscriber with wlrobot_sdk...");

    custom::ChannelFactory::Instance()->Init(1);

    custom::ChannelSubscriber<RealsenseMsg> sub(
        "RealsenseTopic",
        &RealsenseMsg_desc,
        OnRealsenseMsg,
        custom::QoSProfile::VIDEO_STREAM
    );
    sub.InitChannel();

    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return 0;
}

