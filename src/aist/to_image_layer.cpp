#include "to_image_layer.hpp"

const uint32_t code[] = {
#include "aist/to_image.comp.h"
};

vkBasalt::aist::ToImageLayer::ToImageLayer(LogicalDevice *pDevice, VkExtent2D extent2D, uint32_t chainCount)
        : Layer(pDevice, extent2D, chainCount) {}

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

void vkBasalt::aist::ToImageLayer::writeSets(DsWriterHolder holder, uint32_t chainIdx) {
    //Weights are not used, but the common DS needs something written.
    VkWriteDescriptorSet writes[] = {*holder.weights, *holder.outImage, *holder.intermediate};
    writes[0].dstSet = commonDescriptorSet;
    VkDescriptorBufferInfo inIntermediateBufferInfo = *writes[2].pBufferInfo;
    inIntermediateBufferInfo.offset = (imageExtent.width / 2 * (imageExtent.height / 2) * 32) * 4;
    inIntermediateBufferInfo.range = (imageExtent.width * imageExtent.height * 3) * 4;
    writes[2].pBufferInfo = &inIntermediateBufferInfo;
    writes[2].dstSet = writes[1].dstSet = perChainDescriptorSets[chainIdx];
    Layer::writeSets(std::size(writes), writes);
}
