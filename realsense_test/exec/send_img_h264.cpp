#include <chrono>
#include <cstdlib>
#include <cstring>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <wlrobot/robot/channel/channel_publisher.hpp>
#include "RealsenseMsg.h"
#include "utility/logger.h"
namespace custom = wlrobot::robot::channel;

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

rs2::device get_a_realsense_device() {
    rs2::context ctx;
    auto devices = ctx.query_devices();
    if (devices.size() == 0) {
        FRC_ERROR("No device connected!");
        std::exit(0);
    }
    FRC_INFO("get device num: " << devices.size());
    return devices[0];
}

int main() {
    // ---------- RealSense ----------
    auto dev = get_a_realsense_device();
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 15);  // 降到 15 FPS
    pipe.start(cfg);

    const int W = 640, H = 480, FPS = 15;

    // ---------- DDS ----------
    custom::ChannelFactory::Instance()->Init(1);
    custom::ChannelPublisher<RealsenseMsg> pub("RealsenseTopic", &RealsenseMsg_desc, custom::QoSProfile::VIDEO_STREAM);
    pub.InitChannel();

    // ---------- FFmpeg H.264 编码器 ----------
    const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        FRC_ERROR("H.264 codec not found!");
        return -1;
    }

    AVCodecContext* c = avcodec_alloc_context3(codec);
    c->bit_rate = 1000000;               // 1 Mbps (适合 15 FPS)
    c->width = W;
    c->height = H;
    c->time_base = {1, FPS};
    c->framerate = {FPS, 1};
    c->gop_size = 1;                     // 全 I 帧
    c->max_b_frames = 0;                 // 禁用 B 帧
    c->pix_fmt = AV_PIX_FMT_YUV420P;

    // 低延迟参数
    AVDictionary* opts = nullptr;
    av_dict_set(&opts, "preset", "ultrafast", 0);
    av_dict_set(&opts, "tune", "zerolatency", 0);
    av_dict_set(&opts, "x264opts", "repeat-headers=1", 0);

    if (avcodec_open2(c, codec, &opts) < 0) {
        FRC_ERROR("Could not open H.264 codec");
        return -1;
    }

    SwsContext* sws_ctx = sws_getContext(
        W, H, AV_PIX_FMT_BGR24,
        W, H, AV_PIX_FMT_YUV420P,
        SWS_BILINEAR, nullptr, nullptr, nullptr);

    AVFrame* frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_YUV420P;
    frame->width = W;
    frame->height = H;
    av_frame_get_buffer(frame, 32);

    AVPacket* pkt = av_packet_alloc();
    int frame_id = 0;

    // ---------- 主循环 ----------
    using namespace std::chrono;
    auto last_time = steady_clock::now();
    int frame_count = 0;
    size_t total_bytes = 0;

    while (true) {
        rs2::frameset fs = pipe.wait_for_frames();
        rs2::video_frame color = fs.get_color_frame();
        if (!color) continue;

        cv::Mat bgr(cv::Size(W, H), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);

        // --- BGR → YUV420P ---
        uint8_t* in_data[1] = { bgr.data };
        int in_linesize[1] = { 3 * W };
        sws_scale(sws_ctx, in_data, in_linesize, 0, H, frame->data, frame->linesize);
        frame->pts = frame_id++;

        // --- 编码 H.264 ---
        if (avcodec_send_frame(c, frame) < 0) {
            FRC_ERROR("send_frame failed");
            continue;
        }

        while (avcodec_receive_packet(c, pkt) == 0) {
            // --- DDS 消息 ---
            RealsenseMsg msg{};
            msg.width = W;
            msg.height = H;
            msg.timestamp = static_cast<uint64_t>(color.get_timestamp());

            const char* cenc = "h264";
            msg.color_encoding = dds_string_alloc((uint32_t)strlen(cenc));
            memcpy(msg.color_encoding, cenc, strlen(cenc)+1);

            msg.color_frame._length  = msg.color_frame._maximum = pkt->size;
            msg.color_frame._release = true;
            msg.color_frame._buffer  = (uint8_t*)dds_alloc(pkt->size);
            memcpy(msg.color_frame._buffer, pkt->data, pkt->size);

            pub.Write(msg);

            // 带宽统计
            total_bytes += pkt->size;

            size_t raw_bytes = W * H * 3;
            FRC_INFO("[H264] raw=" << raw_bytes/1024 << " KB"
                     << " compressed=" << pkt->size/1024 << " KB"
                     << " ratio=" << (double)pkt->size/raw_bytes*100.0 << "%");

            av_packet_unref(pkt);
        }

        frame_count++;
        auto now = steady_clock::now();
        if (duration_cast<seconds>(now - last_time).count() >= 1) {
            double kbps = (total_bytes / 1024.0); // KB per second
            FRC_HIGHLIGHT("[FPS] " << frame_count << " frames/sec" << " | Bandwidth=" << kbps << " KB/s");
            frame_count = 0;
            total_bytes = 0;
            last_time = now;
        }
    }
    // ---------- 清理 ----------
    avcodec_free_context(&c);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    sws_freeContext(sws_ctx);

    return 0;
}

