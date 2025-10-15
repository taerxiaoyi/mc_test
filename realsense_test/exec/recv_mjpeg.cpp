#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include "RealsenseMsg.h"
#include <opencv2/opencv.hpp>
#include "wlrobot/robot/channel/channel_subscriber.hpp"
#include "utility/logger.h"

namespace custom = wlrobot::robot::channel;

// 检测 JPEG 
static inline bool is_jpeg(const uint8_t* buf, size_t len) {
    return (len >= 4 && buf[0] == 0xFF && buf[1] == 0xD8 && buf[len - 2] == 0xFF && buf[len - 1] == 0xD9);
}

static void draw_label(cv::Mat& img, const std::string& text) {
    if (img.empty()) return;
    int font = cv::FONT_HERSHEY_SIMPLEX;
    double scale = 0.6;
    int thickness = 2;
    cv::putText(img, text, {10, 24}, font, scale, cv::Scalar(255,255,255), thickness, cv::LINE_AA);
    cv::putText(img, text, {10, 24}, font, scale, cv::Scalar(0,0,0), 1, cv::LINE_AA);
}

// =============== 回调函数 ===============
void OnRealsenseMsg(const RealsenseMsg& msg) {
    cv::Mat color;

    if (msg.color_frame._buffer && msg.color_frame._length > 0) {
        const uint8_t* cbuf = msg.color_frame._buffer;
        size_t clen = msg.color_frame._length;
        if (is_jpeg(cbuf, clen)) {
            std::vector<uchar> jpg(cbuf, cbuf + clen);
            color = cv::imdecode(jpg, cv::IMREAD_COLOR);
        } else {
            if (clen >= static_cast<size_t>(msg.width) * msg.height * 3) {
                cv::Mat view(msg.height, msg.width, CV_8UC3, const_cast<uint8_t*>(cbuf));
                color = view.clone();
            }
        }
        if (!color.empty()) draw_label(color, "RGB");
    }

    if (!color.empty()) {
        cv::imshow("Viewer", color);
        cv::waitKey(1);
    }
}

// =============== 主程序 ===============
int main(int argc, char** argv) {
    FRC_INFO("Starting Realsense subscriber with wlrobot_sdk...");
    custom::ChannelFactory::Instance()->Init(2);
    custom::ChannelSubscriber<RealsenseMsg> sub(
        "RealsenseTopic",
        &RealsenseMsg_desc,
        OnRealsenseMsg,
        custom::QoSProfile::VIDEO_STREAM
    );
    sub.InitChannel();

    // 防止退出
    while (true) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}

