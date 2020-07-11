#pragma once

#include "../vulkan_include.hpp"
#include "../logical_device.hpp"

namespace vkBasalt::aist {
    struct NnShader {
        VkShaderModule module = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
    };

    class NnShaders {
    public:
        explicit NnShaders(LogicalDevice *pLogicalDevice);

        virtual ~NnShaders();

        VkDescriptorSetLayout weightsLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout imageLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout bufferLayout = VK_NULL_HANDLE;

        NnShader fromImage;
        NnShader downconvLow;
        NnShader downconvHigh;
        NnShader shuffleLow;
        NnShader shuffleHigh;
        NnShader convMid;
        NnShader in2Dsum;
        NnShader in2Dcoeff;
        NnShader in2Dscale;
        NnShader in2Dfma;
        NnShader upconvHigh;
        NnShader upconvLow;
        NnShader toImage;

        [[nodiscard]] uint32_t getImageAccessColumnGroup() const;

        [[nodiscard]] uint32_t getImageAccessRowGroup() const;

    private:
        LogicalDevice *pLogicalDevice;

        uint32_t imageAccessColumnGroup = 32;

        uint32_t imageAccessRowGroup = 32;

        void createSetLayouts();

        void createSetLayout(VkDescriptorSetLayoutCreateInfo *createInfo, VkDescriptorSetLayout *handle);

        void createShaders();

        void createShader(VkShaderModuleCreateInfo *createInfo, VkShaderModule *handle);

        void createPipelineLayouts();

        void createPipelineLayout(VkPipelineLayoutCreateInfo *createInfo, VkPipelineLayout *handle);

        void createComputePipelines();

        void createComputePipeline(VkComputePipelineCreateInfo *createInfo, VkPipeline *handle);

        void destroyElements(NnShader *holder);

        void destroySetLayout(VkDescriptorSetLayout setLayout);
    };
}


