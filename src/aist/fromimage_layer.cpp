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
        : Layer(pDevice, extent2D, chainCount) {
    depth = 8;
    imageSizeProportion = 8.0;
}

void vkBasalt::aist::FromImageLayer::writeSets(DsWriterHolder holder, uint32_t chainIdx) {
    //No harm in unmodified 2nd intermediate buffer, it's not used. This way API is pleased.
    VkWriteDescriptorSet writes[] = {*holder.weights, *holder.inImage, *holder.intermediate, *holder.intermediate};
    VkDescriptorBufferInfo bufferInfo = *writes[0].pBufferInfo;
    bufferInfo.offset = 0;
    bufferInfo.range = (32 * 3 * 3 * 3 + 32) * 4;
    writes[0].pBufferInfo = &bufferInfo;
    writes[0].dstSet = commonDescriptorSet;
    writes[3].dstSet = writes[2].dstSet = writes[1].dstSet = perChainDescriptorSets[chainIdx];
    writes[3].dstBinding += 1;
    Layer::writeSets(std::size(writes), writes);
}
