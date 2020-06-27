#pragma once

#include "nn_layer.h"

namespace vkBasalt::aist {
    class In2D : public Layer {
    public:
        In2D(LogicalDevice *pDevice, VkExtent2D extent2D, uint32_t chainCount, uint32_t spatialDivisor, uint32_t channels);

        void createLayout(DsCounterHolder *counters) override;

        void createPipeline() override;

        void writeSets(DsWriterHolder holder, uint32_t chainIdx) override;

        void appendCommands(VkCommandBuffer commandBuffer, uint32_t chainIdx,
                            VkBufferMemoryBarrier *bufferBarrierDto) override;

    protected:
        void createPipelineLayout() override;

        const struct Specialization {
            Specialization(VkExtent2D extent2D, uint32_t spatialDivisor, uint32_t channels);

            uint32_t width;
            uint32_t height;
            uint32_t spatialDivisor;
            uint32_t channels;
        } specialization;
    };
}
