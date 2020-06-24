#pragma once

#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>

#include "vulkan_include.hpp"
#include "logical_device.hpp"
#include "config.hpp"
#include "effect.hpp"
#include "aist/nn_layer.h"

namespace vkBasalt
{
    class AistEffect : public Effect
    {
    public:
        AistEffect(LogicalDevice*       pLogicalDevice,
                   VkFormat             format,
                   VkExtent2D           imageExtent,
                   std::vector<VkImage> inputImages,
                   std::vector<VkImage> outputImages,
                   Config*              pConfig);
        virtual void applyEffect(uint32_t imageIndex, VkCommandBuffer commandBuffer) override;
        virtual ~AistEffect();

    private:
        LogicalDevice*               pLogicalDevice;
        VkFormat                     format;
        VkExtent2D                   imageExtent;
        std::vector<VkImage>         inputImages;
        std::vector<VkImage>         outputImages;
        Config*                      pConfig;

        std::vector<VkImageView>     inputImageViews;
        std::vector<VkImageView>     outputImageViews;
        VkDeviceMemory               bufferMemory;
        VkBuffer                     weights;
        std::vector<VkBuffer>        intermediates;
        std::vector<std::unique_ptr<aist::Layer>> layers;
        VkDescriptorPool             descriptorPool;

        void allocateBuffers();
        void relayoutOutputImages();
        void createLayoutAndDescriptorSets();
    };

} // namespace vkBasalt