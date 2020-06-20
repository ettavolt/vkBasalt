#ifndef EFFECT_AIST_HPP_INCLUDED
#define EFFECT_AIST_HPP_INCLUDED
#include <vector>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <memory>

#include "vulkan_include.hpp"

#include "effect.hpp"
#include "config.hpp"

#include "logical_device.hpp"

namespace vkBasalt
{
    class AistEffect : public Effect
    {
    public:
        AistEffect(LogicalDevice*       pLogicalDevice,
                   VkFormat             format,
                   std::vector<VkImage> inputImages,
                   std::vector<VkImage> outputImages,
                   Config*              pConfig);
        virtual void applyEffect(uint32_t imageIndex, VkCommandBuffer commandBuffer) override;
        virtual ~AistEffect();

    protected:
        LogicalDevice*               pLogicalDevice;
        std::vector<VkImage>         inputImages;
        std::vector<VkImage>         outputImages;
        std::vector<VkImageView>     inputImageViews;
        std::vector<VkImageView>     outputImageViews;
        VkDescriptorSetLayout        descriptorSetLayout;
        VkDescriptorPool             descriptorPool;
        std::vector<VkDescriptorSet> imageDescriptorSets;
        VkShaderModule               computeModule;
        VkPipelineLayout             pipelineLayout;
        VkPipeline                   computePipeline;
        VkFormat                     format;
        Config*                      pConfig;
        std::vector<uint32_t>        computeCode;

    private:
        void relayoutOutputImages();
        void createLayoutAndDescriptorSets();
    };
} // namespace vkBasalt

#endif // EFFECT_AIST_HPP_INCLUDED
