#include "fromimage_layer.hpp"

const uint32_t code[] = {
#include "aist/from_image.comp.h"
};

void vkBasalt::aist::FromImageLayer::createLayout(DsCounterHolder *counters) {
    Layer::createLayout(true);
    counters->images += chainCount;
    counters->uniforms++;
    counters->intermediates += chainCount;
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

void vkBasalt::aist::FromImageLayer::writeSets(DsWriterHolder holder, uint32_t chainIdx) {
    VkWriteDescriptorSet writes[] = {*holder.weights, *holder.inImage, *holder.intermediate};
    VkDescriptorBufferInfo bufferInfo = *writes[0].pBufferInfo;
    bufferInfo.offset = 0;
    bufferInfo.range = 10 * 3 * 4;// + 8; //8 for 16-bytes alignment.
    writes[0].pBufferInfo = &bufferInfo;
    writes[0].dstSet = commonDescriptorSet;
    writes[2].dstSet = writes[1].dstSet = perChainDescriptorSets[chainIdx];
    Layer::writeSets(std::size(writes), writes);
}
