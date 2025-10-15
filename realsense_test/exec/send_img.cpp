#include <cstdlib>
#include <cstring>
#include <chrono>
#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <wlrobot/robot/channel/channel_publisher.hpp>
namespace custom = wlrobot::robot::channel;

#include "RealsenseMsg.h"
#include "utility/logger.h"

// check if camera connected and return the first one
rs2::device get_a_realsense_device(){
    rs2::context ctx;
    rs2::device_list devices = ctx.query_devices();
    if (devices.size() == 0){
        FRC_ERROR("No device connected!");
        std::exit(0);
    }
    FRC_INFO("get device num: "<<devices.size());
    for (int i = 0; i < devices.size(); i++){
        std::vector<rs2::sensor> sensors = devices[i].query_sensors();
        std::string serial = devices[i].get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
        std::string port = devices[i].get_info(RS2_CAMERA_INFO_PHYSICAL_PORT);
        FRC_INFO("Device number " << i+1 << ", serial" << " : " << serial << ", port " << " : " << port);
    }
    return devices[0];
}

int main() {
    rs2::device selected_device;
    selected_device = get_a_realsense_device();
    std::string serial = selected_device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);  // get serial number

    rs2::config cfg;
    cfg.enable_device(serial);
    int fps = 15;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, fps);

    rs2::pipeline pipe;
    pipe.start(cfg);

    // create dds
    custom::ChannelFactory::Instance()->Init(2);
    custom::ChannelPublisher<RealsenseMsg> pub("RealsenseTopic", &RealsenseMsg_desc, custom::QoSProfile::VIDEO_STREAM);
    pub.InitChannel();

    const int JPEG_COMPRESSION_QUALITY = 30;            
    FRC_HIGHLIGHT("[main] RGB(JPEG compression quality="<<JPEG_COMPRESSION_QUALITY<<")\n");
    
    using namespace std::chrono;
    auto last_time = steady_clock::now();
    int frame_count = 0;

    while (true) {
        rs2::frameset fs = pipe.wait_for_frames();
        rs2::video_frame color = fs.get_color_frame(); // get color frame

        const uint32_t W = color.get_width();
        const uint32_t H = color.get_height();

        // --- 彩色：BGR8 -> JPEG ---
        cv::Mat bgr(cv::Size(W, H), CV_8UC3, const_cast<void*>(color.get_data()), cv::Mat::AUTO_STEP);

        std::vector<uchar> color_jpeg;
        std::vector<int> jpg_params = { cv::IMWRITE_JPEG_QUALITY, JPEG_COMPRESSION_QUALITY };
        if (!cv::imencode(".jpg", bgr, color_jpeg, jpg_params)) {
            std::cerr << "imencode(.jpg) failed\n"; 
            continue;
        }

        // --- 填充并发送 DDS 消息 ---
        RealsenseMsg msg{};
        msg.width     = W;
        msg.height    = H;
        msg.timestamp = static_cast<uint64_t>(color.get_timestamp()); // 也可用 now_ns()

        // color
        const char* cenc = "jpeg";
        msg.color_encoding = dds_string_alloc((uint32_t)std::strlen(cenc));
        std::memcpy(msg.color_encoding, cenc, std::strlen(cenc)+1);

        msg.color_frame._length  = msg.color_frame._maximum = (uint32_t)color_jpeg.size();
        msg.color_frame._release = true;
        msg.color_frame._buffer  = (uint8_t*) dds_alloc(color_jpeg.size());
        std::memcpy(msg.color_frame._buffer, color_jpeg.data(), color_jpeg.size());
        pub.Write(msg);

        size_t raw_color_bytes = W * H * 3;    // BGR8, 3 channels, each pixel 1 byte, so one frame is W*H*3 bytes         
        size_t jpeg_bytes  = color_jpeg.size();
        if (frame_count % fps == 0){ // each fps frames print once
            FRC_INFO("[Frame] "
                    << W << "x" << H
                    << " raw_color=" << raw_color_bytes/1024 << " KB"
                    << " jpeg=" << jpeg_bytes/1024 << " KB");
        }

        frame_count++; 
        auto now = steady_clock::now();  // calculate and print fps
        if (duration_cast<seconds>(now - last_time).count() >= 1) { 
            FRC_HIGHLIGHT("[FPS] " << frame_count << " frames/sec"); 
            frame_count = 0; 
            last_time = now; 
        }
    }
    return 0;
}
