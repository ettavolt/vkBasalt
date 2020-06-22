#include "fromimage_layer.hpp"

const uint32_t code[] = {
#include "aist/from_image.comp.h"
};

void vkBasalt::aist::FromImageLayer::createLayout() {
    Layer::createLayout(true);
}

void vkBasalt::aist::FromImageLayer::createPipeline() {
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

vkBasalt::aist::FromImageLayer::FromImageLayer(LogicalDevice *pDevice, VkExtent2D extent2D, uint32_t chainCount)
        : Layer(pDevice, extent2D, chainCount) {}

void vkBasalt::aist::FromImageLayer::writeSets(
        VkWriteDescriptorSet *inImage,
        VkWriteDescriptorSet *outImage,
        uint32_t chainIdx
) {
    inImage->dstBinding = 1;
    inImage->dstSet = perChainDescriptorSets[chainIdx];
    Layer::writeSets(1, inImage);
}
