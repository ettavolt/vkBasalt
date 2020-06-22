#pragma once

#include "nn_layer.h"

namespace vkBasalt::aist {
    class FromImageLayer : public Layer {
    public:
        FromImageLayer(LogicalDevice *pDevice, VkExtent2D extent2D, uint32_t chainCount);

        void createLayout() override;

        void writeSets(VkWriteDescriptorSet *inImage, VkWriteDescriptorSet *outImage, uint32_t chainIdx) override;

        void createPipeline() override;

     };
}