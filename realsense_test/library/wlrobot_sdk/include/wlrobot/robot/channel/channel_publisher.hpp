#pragma once
#include "channel_factory.hpp"

namespace wlrobot::robot::channel
{

template<typename MSG>
class ChannelPublisher
{
public:
    ChannelPublisher(const std::string& name,
                     const dds_topic_descriptor* desc,
                     QoSProfile profile = QoSProfile::DEFAULT)
        : name_(name), desc_(desc), profile_(profile), writer_(0) {}

    void InitChannel()
    {
        writer_ = ChannelFactory::Instance()->CreateSendChannel<MSG>(name_, desc_, profile_);
    }

    bool Write(const MSG& msg, int64_t /*waitMicrosec*/ = 0)
    {
        if (writer_ > 0)
        {
            return dds_write(writer_, &msg) >= 0;
        }
        return false;
    }

private:
    std::string name_;
    const dds_topic_descriptor* desc_;
    QoSProfile profile_;
    dds_entity_t writer_;
};

} // namespace wlrobot::robot::channel

