#pragma once

#include "../vulkan_include.hpp"
#include "../logical_device.hpp"

namespace vkBasalt::aist {
    struct NnShader {
        VkShaderModule module = VK_NULL_HANDLE;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        const uint32_t scale = 0xFFFFu;
        const uint32_t depthGlobals = 1u;
    };

    const uint32_t IMAGE_CHANNELS = 3u;
    const uint32_t LOW_STRIDED_CHANNELS = 15u;
    const uint32_t LOW_SHUFFLE_CHANNELS = 16u;
    const uint32_t HIGH_CHANNELS = 64u;

    class NnShaders {
    public:
        explicit NnShaders(LogicalDevice *pLogicalDevice);

        virtual ~NnShaders();

        VkDescriptorSetLayout weightsLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout imageLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout bufferLayout = VK_NULL_HANDLE;

        NnShader fromImage{.scale = 32u};
        NnShader downConvLow{.scale = 8u, .depthGlobals = IMAGE_CHANNELS};
        NnShader downConvHigh{.scale = 2u, .depthGlobals = LOW_SHUFFLE_CHANNELS};
        NnShader downShuffleLow{.scale = 64u};
        NnShader shuffleHigh;
        NnShader convHigh;
        NnShader in2Dsum{.scale = 1024u, .depthGlobals = LOW_SHUFFLE_CHANNELS};
        NnShader in2Dcoeff{.scale = 1024u, .depthGlobals = LOW_SHUFFLE_CHANNELS};
        NnShader in2Dscale{.scale = 1024u / LOW_SHUFFLE_CHANNELS, .depthGlobals = 1u};
        NnShader in2DscaleAndAdd;
        NnShader upConvHigh{.scale = 4u, .depthGlobals = LOW_SHUFFLE_CHANNELS};
        //Need different spec constant to specify direction.
        NnShader upShuffleLow{.scale = 64u};
        NnShader upConvLow{.scale = 16u, .depthGlobals = IMAGE_CHANNELS};
        NnShader toImage{.scale = 32u};

    private:
        LogicalDevice *pLogicalDevice;
        VkPipelineLayout imageBufferWidthHeightLayout = VK_NULL_HANDLE;
        VkPipelineLayout twoBufferWeightsWidthHeightLayout = VK_NULL_HANDLE;

        void createSetLayouts();

        void createSetLayout(VkDescriptorSetLayoutCreateInfo *createInfo, VkDescriptorSetLayout *handle);

        void createShaders();

        void createShader(VkShaderModuleCreateInfo *createInfo, VkShaderModule *handle);

        void createPipelineLayouts();

        void createPipelineLayout(VkPipelineLayoutCreateInfo *createInfo, VkPipelineLayout *handle);

        void createComputePipelines();

        void createComputePipeline(VkComputePipelineCreateInfo *createInfo, NnShader *shader);

        void destroyElements(NnShader *holder);

        void destroySetLayout(VkDescriptorSetLayout setLayout);
    };
}


