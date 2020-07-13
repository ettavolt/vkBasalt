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
#include "aist/nn_shaders.h"

namespace vkBasalt {
    class AistEffect : public Effect {
    public:
        AistEffect(
            LogicalDevice *pLogicalDevice,
            VkFormat format,
            VkExtent2D imageExtent,
            std::vector<VkImage> inputImages,
            std::vector<VkImage> outputImages,
            Config *pConfig
        );

        void applyEffect(uint32_t imageIndex, VkCommandBuffer commandBuffer) override;

        ~AistEffect() override;

    private:
        LogicalDevice *pLogicalDevice;
        VkFormat format;
        VkExtent2D imageExtent;
        VkExtent2D lowExtent;
        VkExtent2D highExtent;
        std::vector<VkImage> inputImages;
        std::vector<VkImage> outputImages;
        Config *pConfig;

        //TODO: bind this to pLogicalDevice life
        aist::NnShaders shaders;

        const VkDeviceSize intermediateChunkAlignedSize;
        std::vector<VkImageView> inputImageViews;
        std::vector<VkImageView> outputImageViews;
        VkDeviceMemory bufferMemory = VK_NULL_HANDLE;
        VkBuffer weights = VK_NULL_HANDLE;
        std::vector<VkBuffer> intermediates;
        VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
        VkDescriptorSet wStridedLowDescriptorSet = VK_NULL_HANDLE;
        VkDescriptorSet wShuffleLowDescriptorSet = VK_NULL_HANDLE;
        VkDescriptorSet wNormLowDescriptorSet = VK_NULL_HANDLE;
        VkDescriptorSet wConvHighDescriptorSet = VK_NULL_HANDLE;
        VkDescriptorSet wShuffleHighDescriptorSet = VK_NULL_HANDLE;
        VkDescriptorSet wNormHighDescriptorSet = VK_NULL_HANDLE;
        std::vector<VkDescriptorSet> inputImageDescriptorSets;
        std::vector<VkDescriptorSet> outputImageDescriptorSets;
        std::vector<VkDescriptorSet> startThirdBufferDescriptorSets;
        std::vector<VkDescriptorSet> midThirdBufferDescriptorSets;
        std::vector<VkDescriptorSet> endThirdBufferDescriptorSets;

        void allocateBuffers();

        void relayoutOutputImages();

        void createDescriptorSets();

        void appendBufferBarrier(VkCommandBuffer commandBuffer, VkBufferMemoryBarrier *bufferBarrierDto);

        void dispatchImageTouching(VkCommandBuffer commandBuffer, uint32_t chainIdx, bool output);

        void dispatchStridedLow(
            VkCommandBuffer commandBuffer,
            uint32_t chainIdx,
            uint32_t weightsOffset,
            bool up
        );

        void dispatchLinearLowMidEnd(
            VkCommandBuffer commandBuffer,
            uint32_t chainIdx,
            uint32_t weightsOffset,
            bool inFromMid,
            vkBasalt::aist::NnShader &shader
        );

        void dispatchIn2D(
            VkCommandBuffer commandBuffer,
            VkBufferMemoryBarrier *bufferBarrierDto,
            uint32_t chainIdx,
            uint32_t weightsOffset
        );

        static VkDeviceSize alignTo256Bytes(VkDeviceSize size);
    };

} // namespace vkBasalt