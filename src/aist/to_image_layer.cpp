#include "to_image_layer.hpp"

const uint32_t code[] = {
#include "aist/to_image.comp.h"
};

void vkBasalt::aist::ToImageLayer::createLayout(DsCounterHolder *counters) {
    Layer::createLayout(true);
    counters->images += chainCount;
    counters->uniforms++;
    counters->intermediates += chainCount;
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
        : Layer(pDevice, extent2D, chainCount) {
    imageSizeProportion = 8.0;
}

void vkBasalt::aist::ToImageLayer::writeSets(DsWriterHolder holder, uint32_t chainIdx) {
    VkWriteDescriptorSet writes[] = {*holder.weights, *holder.outImage, *holder.intermediate, *holder.intermediate};
    VkDescriptorBufferInfo weightsBufferInfo = *writes[0].pBufferInfo;
    weightsBufferInfo.offset = (32 * 3 * 3 * 3 + 32) * 4;
    weightsBufferInfo.range = (32 * 3 * 3 * 3 + 3) * 4;
    writes[0].pBufferInfo = &weightsBufferInfo;
    writes[0].dstSet = commonDescriptorSet;
    writes[3].dstSet = writes[2].dstSet = writes[1].dstSet = perChainDescriptorSets[chainIdx];
    writes[3].dstBinding += 1;
    VkDescriptorBufferInfo inIntermediateBufferInfo = *writes[3].pBufferInfo;
    inIntermediateBufferInfo.range = (imageExtent.width / 2 * (imageExtent.height / 2) * 32) * 4;
    writes[2].pBufferInfo = &inIntermediateBufferInfo;
    VkDescriptorBufferInfo subIntermediateBufferInfo = *writes[3].pBufferInfo;
    subIntermediateBufferInfo.offset = inIntermediateBufferInfo.range;
    subIntermediateBufferInfo.range = (imageExtent.width * imageExtent.height * 3) * 4;
    writes[3].pBufferInfo = &subIntermediateBufferInfo;
    Layer::writeSets(std::size(writes), writes);
}
