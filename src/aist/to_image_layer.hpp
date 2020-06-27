#pragma once

#include "nn_layer.h"

namespace vkBasalt::aist {
    class ToImageLayer : public Layer {
    public:
        ToImageLayer(LogicalDevice *pDevice, VkExtent2D extent2D, uint32_t chainCount);

        void createLayout(DsCounterHolder *counters) override;

        void writeSets(DsWriterHolder holder, uint32_t chainIdx) override;

        void createPipeline() override;

        void appendCommands(VkCommandBuffer commandBuffer, uint32_t chainIdx,
                            VkBufferMemoryBarrier *bufferBarrierDto) override;

    protected:
        void createPipelineLayout() override;

    };
}