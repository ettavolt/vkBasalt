#pragma once

#include "../vulkan_include.hpp"
#include "../logical_device.hpp"

namespace vkBasalt::aist {
    struct DsCounterHolder {
        uint32_t images;
        uint32_t uniforms;
        uint32_t intermediates;
    };
    struct DsWriterHolder {
        VkWriteDescriptorSet *inImage;
        VkWriteDescriptorSet *outImage;
        VkWriteDescriptorSet *intermediate;
        VkWriteDescriptorSet *weights;
    };

    class Layer {
    public:
        static VkDeviceSize alignTo256Bytes(VkDeviceSize size);

        virtual void createLayout(DsCounterHolder *counters) = 0;

        virtual void writeSets(DsWriterHolder holder, uint32_t chainIdx) = 0;

        virtual void createDescriptorSets(VkDescriptorPool descriptorPool);

        virtual void createPipeline();

        virtual void appendCommands(
                VkCommandBuffer commandBuffer,
                uint32_t chainIdx,
                VkBufferMemoryBarrier *bufferBarrierDto
        );

        virtual ~Layer();

    protected:
        Layer(LogicalDevice *pLogicalDevice, VkExtent2D imageExtent, uint32_t chainCount);

        LogicalDevice *pLogicalDevice;
        VkExtent2D imageExtent;
        uint32_t chainCount;
        float imageSizeProportion = 16.0;
        uint32_t depth = 1;
        VkDescriptorSetLayout commonDescriptorSetLayout = nullptr;
        VkDescriptorSetLayout perChainDescriptorSetLayout = nullptr;
        VkDescriptorSet commonDescriptorSet = nullptr;
        std::vector<VkDescriptorSet> perChainDescriptorSets;
        VkShaderModule computeModule = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkPipeline computePipeline = VK_NULL_HANDLE;

        void createLayout(bool tapsIntoImage);

        virtual void createPipelineLayout();

        void writeSets(uint32_t count, VkWriteDescriptorSet *writes);

        void dispatchPipeline(VkCommandBuffer commandBuffer, uint32_t chainIdx);
    };
}