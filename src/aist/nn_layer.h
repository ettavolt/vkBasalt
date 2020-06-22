#pragma once

#include "../vulkan_include.hpp"
#include "../logical_device.hpp"

namespace vkBasalt::aist {
    class Layer {
    public:
        VkDescriptorSetLayout commonDescriptorSetLayout = nullptr;
        VkDescriptorSetLayout perChainDescriptorSetLayout = nullptr;
        VkDescriptorSet commonDescriptorSet = nullptr;
        std::vector<VkDescriptorSet> perChainDescriptorSets;
        VkShaderModule computeModule = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline computePipeline = VK_NULL_HANDLE;

        virtual void createLayout() = 0;

        virtual void writeSets(VkWriteDescriptorSet *inImage, VkWriteDescriptorSet *outImage, uint32_t chainIdx) = 0;

        virtual void createPipeline();

        virtual ~Layer();

    protected:
        Layer(LogicalDevice *pLogicalDevice, VkExtent2D imageExtent, uint32_t chainCount);;

        LogicalDevice *pLogicalDevice;
        VkExtent2D imageExtent;
        uint32_t chainCount;

        void createLayout(bool tapsIntoImage);

        void writeSets(uint32_t count, VkWriteDescriptorSet *writes);
    };
}