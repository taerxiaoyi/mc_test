#pragma once
#include <dds/dds.h>
#include <string>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <iostream>

namespace wlrobot::robot::channel
{

class ChannelFactory
{
public:
    static ChannelFactory* Instance()
    {
        static ChannelFactory inst;
        return &inst;
    }

    void Init(int32_t domainId = 0)
    {
        std::lock_guard<std::mutex> lk(mtx_);
        if (inited_) return;

        participant_ = dds_create_participant(domainId, nullptr, nullptr);
        if (participant_ < 0)
        {
            throw std::runtime_error("[ChannelFactory] Failed to create CycloneDDS participant");
        }
        inited_ = true;
    }

    template<typename MSG>
    dds_entity_t CreateSendChannel(const std::string& name, const dds_topic_descriptor* desc)
    {
        dds_entity_t topic = dds_create_topic(participant_, desc, name.c_str(), nullptr, nullptr);
        if (topic < 0) throw std::runtime_error("[ChannelFactory] Failed to create topic for " + name);

        dds_entity_t writer = dds_create_writer(participant_, topic, nullptr, nullptr);
        if (writer < 0) throw std::runtime_error("[ChannelFactory] Failed to create writer for " + name);

        return writer;
    }

    template<typename MSG>
    dds_entity_t CreateRecvChannel(const std::string& name,
                                   const dds_topic_descriptor* desc,
                                   std::function<void(const MSG&)> cb)
    {
        dds_entity_t topic = dds_create_topic(participant_, desc, name.c_str(), nullptr, nullptr);
        if (topic < 0) throw std::runtime_error("[ChannelFactory] Failed to create topic for " + name);

        dds_entity_t reader = dds_create_reader(participant_, topic, nullptr, nullptr);
        if (reader < 0) throw std::runtime_error("[ChannelFactory] Failed to create reader for " + name);
        
        // ---- 等待 Publisher ----
        dds_set_status_mask(reader, DDS_SUBSCRIPTION_MATCHED_STATUS);
        std::cout << "[ChannelSubscriber] Waiting for publisher on " << name << " ..." << std::endl;
        while (true) {
            uint32_t status = 0;
            (void)dds_get_status_changes(reader, &status);
            if (status & DDS_SUBSCRIPTION_MATCHED_STATUS) {
                std::cout << "[ChannelSubscriber] Publisher discovered for " << name << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // 开线程轮询消息
        std::thread([reader, cb]() {
            while (true)
            {
                void* samples[1];
                dds_sample_info_t infos[1];
                int n = dds_take(reader, samples, infos, 1, 1);
                if (n > 0 && infos[0].valid_data)
                {
                    MSG* m = reinterpret_cast<MSG*>(samples[0]);
                    cb(*m);
                    dds_return_loan(reader, samples, n);
                }
                dds_sleepfor(DDS_MSECS(10));
            }
        }).detach();

        return reader;
    }


private:
    ChannelFactory() : inited_(false), participant_(0) {}
    bool inited_;
    dds_entity_t participant_;
    std::mutex mtx_;
};

} // namespace wlrobot::robot::channel

