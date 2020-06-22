#include "to_image_layer.hpp"

const uint32_t code[] = {
#include "aist/to_image.comp.h"
};

void vkBasalt::aist::ToImageLayer::createLayout() {
    Layer::createLayout(true);
}

void vkBasalt::aist::ToImageLayer::createPipeline() {
    VkShaderModuleCreateInfo shaderCreateInfo{
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .pNext = nullptr,
            .flags = 0,
            .codeSize = sizeof(code), .pCode = code
    };
    VkResult result = pLogicalDevice->vkd.CreateShaderModule(
            pLogicalDevice->device,
            &shaderCreateInfo,
            nullptr,
            &computeModule
    );
    ASSERT_VULKAN(result)
    Layer::createPipeline();
}

vkBasalt::aist::ToImageLayer::ToImageLayer(LogicalDevice *pDevice, VkExtent2D extent2D, uint32_t chainCount)
        : Layer(pDevice, extent2D, chainCount) {}

void vkBasalt::aist::ToImageLayer::writeSets(
        VkWriteDescriptorSet *inImage,
        VkWriteDescriptorSet *outImage,
        uint32_t chainIdx
) {
    outImage->dstBinding = 1;
    outImage->dstSet = perChainDescriptorSets[chainIdx];
    Layer::writeSets(1, outImage);
}
