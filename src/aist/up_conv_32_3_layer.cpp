#include <gmpxx.h>
#include <cmath>
#include "up_conv_32_3_layer.hpp"

const uint32_t code[] = {
#include "aist/up_conv_32_3.comp.h"
};

void vkBasalt::aist::UpConv32t3::createLayout(DsCounterHolder *counters) {
    Layer::createLayout(false);
    counters->uniforms++;
    counters->intermediates += chainCount * 2;
}

vkBasalt::aist::UpConv32t3::UpConv32t3(LogicalDevice *pDevice, VkExtent2D extent2D, uint32_t chainCount)
        : Layer(pDevice, extent2D, chainCount) {
    imageSizeProportion = 8.0;
}

void vkBasalt::aist::UpConv32t3::createPipeline() {
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

void vkBasalt::aist::UpConv32t3::createPipelineLayout() {
    VkDescriptorSetLayout setLayouts[]{
            commonDescriptorSetLayout,
            perChainDescriptorSetLayout,
    };
    VkPushConstantRange pushConstantRange {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t),
    };
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = std::size(setLayouts),
            .pSetLayouts = setLayouts,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange,
    };
    VkResult result = pLogicalDevice->vkd.CreatePipelineLayout(
            pLogicalDevice->device,
            &pipelineLayoutCreateInfo,
            nullptr,
            &pipelineLayout
    );
    ASSERT_VULKAN(result)
}

void vkBasalt::aist::UpConv32t3::writeSets(DsWriterHolder holder, uint32_t chainIdx) {
    VkWriteDescriptorSet writes[] = {*holder.weights, *holder.intermediate, *holder.intermediate};
    VkDescriptorBufferInfo weightsBufferInfo = *writes[0].pBufferInfo;
    weightsBufferInfo.offset = (32 * 3 * 3 * 3 + 32) * 4;
    weightsBufferInfo.range = (32 * 3 * 3 * 3 + 3) * 4;
    writes[0].pBufferInfo = &weightsBufferInfo;
    writes[0].dstSet = commonDescriptorSet;
    writes[2].dstSet = writes[1].dstSet = perChainDescriptorSets[chainIdx];
    VkDescriptorBufferInfo inIntermediateBufferInfo = *writes[1].pBufferInfo;
    inIntermediateBufferInfo.range = (imageExtent.width / 2 * (imageExtent.height / 2) * 32) * 4;
    writes[1].pBufferInfo = &inIntermediateBufferInfo;
    writes[1].dstBinding = 0;
    VkDescriptorBufferInfo subIntermediateBufferInfo = *writes[2].pBufferInfo;
    subIntermediateBufferInfo.offset = inIntermediateBufferInfo.range;
    subIntermediateBufferInfo.range = (imageExtent.width * imageExtent.height * 3) * 4;
    writes[2].pBufferInfo = &subIntermediateBufferInfo;
    Layer::writeSets(std::size(writes), writes);
}

void vkBasalt::aist::UpConv32t3::appendCommands(VkCommandBuffer commandBuffer, uint32_t chainIdx,
                                                VkBufferMemoryBarrier *bufferBarrierDto) {
    bool addBufferBarrier = false;
    for (uint32_t substage = 0; substage < 4; substage++) {
        if (addBufferBarrier) {
            pLogicalDevice->vkd.CmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0, nullptr,
                    1, bufferBarrierDto,
                    0, nullptr
            );
        }
        pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        VkDescriptorSet sets[]{
                commonDescriptorSet,
                perChainDescriptorSets[chainIdx],
        };
        pLogicalDevice->vkd.CmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                pipelineLayout, 0,
                std::size(sets), sets,
                0, nullptr
        );
        pLogicalDevice->vkd.CmdPushConstants(
                commandBuffer,
                pipelineLayout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0,
                sizeof(substage),
                &substage
        );
        pLogicalDevice->vkd.CmdDispatch(
                commandBuffer,
                (uint32_t) std::ceil(imageExtent.width / imageSizeProportion),
                (uint32_t) std::ceil(imageExtent.height / imageSizeProportion),
                depth
        );
        addBufferBarrier = true;
    }
}
