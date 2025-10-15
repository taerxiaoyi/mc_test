#pragma once
#include "channel_factory.hpp"

namespace wlrobot::robot::channel
{

template<typename MSG>
class ChannelSubscriber
{
public:
    ChannelSubscriber(const std::string& name,
                      const dds_topic_descriptor* desc,
                      std::function<void(const MSG&)> cb)
        : name_(name), desc_(desc), cb_(cb), reader_(0) {}

    void InitChannel()
    {
        reader_ = ChannelFactory::Instance()->CreateRecvChannel<MSG>(name_, desc_, cb_);
    }

private:
    std::string name_;
    const dds_topic_descriptor* desc_;
    std::function<void(const MSG&)> cb_;
    dds_entity_t reader_;
};

} // namespace wlrobot::robot::channel

