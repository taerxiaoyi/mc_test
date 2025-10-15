cmake_minimum_required(VERSION 3.14)
project(get_camera C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- 依赖 ----
find_package(OpenCV REQUIRED)
find_package(realsense2 QUIET)

# ---- FFmpeg (libavcodec/libavutil/libswscale) ----
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavcodec
    libavutil
    libswscale
)

set(IDL_GENERATED_DIR   ${CMAKE_SOURCE_DIR}/msg)
set(WESTLAKE_SDK        ${CMAKE_SOURCE_DIR}/library/wlrobot_sdk)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${WESTLAKE_SDK}/include
    ${IDL_GENERATED_DIR}
)

add_subdirectory(library/robot_shared)

set(COMMON_LIBS
    pthread
    robot_shared_cyclonedds
    # robot_shared_westlake_cyclonedds_sdk
)

# recv_mjpeg
add_executable(recv_mjpeg
    exec/recv_mjpeg.cpp
    ${IDL_GENERATED_DIR}/RealsenseMsg.c
)
target_link_libraries(recv_mjpeg
    ${OpenCV_LIBS}
    ${COMMON_LIBS}
)

# recv_h264
add_executable(recv_h264
    exec/recv_h264.cpp
    ${IDL_GENERATED_DIR}/RealsenseMsg.c
)
target_link_libraries(recv_h264
    ${COMMON_LIBS}
    ${OpenCV_LIBS}
    PkgConfig::FFMPEG
)

# send_img
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    message(STATUS "Detected aarch64 → force using manual realsense2 path")

    set(REALSENSE2_INCLUDE_DIR /usr/include/librealsense2)
    set(REALSENSE2_LIBRARY /usr/lib/aarch64-linux-gnu/librealsense2.so)

    include_directories(${REALSENSE2_INCLUDE_DIR})

    add_executable(send_img
        exec/send_img.cpp
        ${IDL_GENERATED_DIR}/RealsenseMsg.c
    )
    target_link_libraries(send_img
        ${REALSENSE2_LIBRARY}
        ${COMMON_LIBS}
        ${OpenCV_LIBS}
    )

    add_executable(send_img_h264
        exec/send_img_h264.cpp
        ${IDL_GENERATED_DIR}/RealsenseMsg.c
    )

    target_link_libraries(send_img_h264
        realsense2::realsense2
        ${COMMON_LIBS}
        ${OpenCV_LIBS}
        PkgConfig::FFMPEG
    )


elseif (realsense2_FOUND)
    message(STATUS "x86 detected realsense2 via CMake package")

    add_executable(send_img
        exec/send_img.cpp
        ${IDL_GENERATED_DIR}/RealsenseMsg.c
    )
    target_link_libraries(send_img
        realsense2::realsense2
        ${COMMON_LIBS}
        ${OpenCV_LIBS}
    )

    add_executable(send_img_h264
        exec/send_img_h264.cpp
        ${IDL_GENERATED_DIR}/RealsenseMsg.c
    )

    target_link_libraries(send_img_h264
        realsense2::realsense2
        ${COMMON_LIBS}
        ${OpenCV_LIBS}
        PkgConfig::FFMPEG
    )

else()
    message(WARNING "x86 but realsense2 not found → skipping send_img")
endif()

