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

enum class QoSProfile {
    DEFAULT,        // CycloneDDS 默认 (RELIABLE, KEEP_LAST=1000)
    VIDEO_STREAM,   // BEST_EFFORT, KEEP_LAST=1
    SENSOR_FAST,    // BEST_EFFORT, KEEP_LAST=10
    RELIABLE_STATE  // RELIABLE, KEEP_LAST=100
};

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
    dds_entity_t CreateSendChannel(const std::string& name,
                                   const dds_topic_descriptor* desc,
                                   QoSProfile qos_profile = QoSProfile::DEFAULT)
    {
        dds_entity_t topic = dds_create_topic(participant_, desc, name.c_str(), nullptr, nullptr);
        if (topic < 0) throw std::runtime_error("[ChannelFactory] Failed to create topic for " + name);

        dds_qos_t* qos = make_qos(qos_profile);
        dds_entity_t writer = dds_create_writer(participant_, topic, qos, nullptr);
        dds_delete_qos(qos);

        if (writer < 0) throw std::runtime_error("[ChannelFactory] Failed to create writer for " + name);
        return writer;
    }

    template<typename MSG>
    dds_entity_t CreateRecvChannel(const std::string& name,
                                   const dds_topic_descriptor* desc,
                                   std::function<void(const MSG&)> cb,
                                   QoSProfile qos_profile = QoSProfile::DEFAULT)
    {
        dds_entity_t topic = dds_create_topic(participant_, desc, name.c_str(), nullptr, nullptr);
        if (topic < 0) throw std::runtime_error("[ChannelFactory] Failed to create topic for " + name);

        dds_qos_t* qos = make_qos(qos_profile);
        dds_entity_t reader = dds_create_reader(participant_, topic, qos, nullptr);
        dds_delete_qos(qos);

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
                dds_sleepfor(DDS_MSECS(5));
            }
        }).detach();

        return reader;
    }

private:
    ChannelFactory() : inited_(false), participant_(0) {}

    dds_qos_t* make_qos(QoSProfile profile) {
        dds_qos_t* qos = dds_create_qos();
        switch (profile) {
            case QoSProfile::VIDEO_STREAM:
                dds_qset_reliability(qos, DDS_RELIABILITY_BEST_EFFORT, DDS_SECS(0));
                dds_qset_history(qos, DDS_HISTORY_KEEP_LAST, 1);
                dds_qset_durability(qos, DDS_DURABILITY_VOLATILE);
                dds_qset_resource_limits(qos, 1, DDS_LENGTH_UNLIMITED, DDS_LENGTH_UNLIMITED);
                break;
            case QoSProfile::SENSOR_FAST:
                dds_qset_reliability(qos, DDS_RELIABILITY_BEST_EFFORT, DDS_SECS(0));
                dds_qset_history(qos, DDS_HISTORY_KEEP_LAST, 10);
                dds_qset_durability(qos, DDS_DURABILITY_VOLATILE);
                dds_qset_resource_limits(qos, 10, DDS_LENGTH_UNLIMITED, DDS_LENGTH_UNLIMITED);
                break;
            case QoSProfile::RELIABLE_STATE:
                dds_qset_reliability(qos, DDS_RELIABILITY_RELIABLE, DDS_SECS(1));
                dds_qset_history(qos, DDS_HISTORY_KEEP_LAST, 100);
                break;
            case QoSProfile::DEFAULT:
            default:
                // CycloneDDS 默认
                break;
        }
        return qos;
    }

    bool inited_;
    dds_entity_t participant_;
    std::mutex mtx_;
};

} // namespace wlrobot::robot::channel

