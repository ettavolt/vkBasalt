#pragma once

#include "nn_layer.h"

namespace vkBasalt::aist {
    class FromImageLayer : public Layer {
    public:
        FromImageLayer(LogicalDevice *pDevice, VkExtent2D extent2D, uint32_t chainCount);

        void createLayout(DsCounterHolder *counters) override;

        void writeSets(DsWriterHolder holder, uint32_t chainIdx) override;

        void createPipeline() override;

     };
}