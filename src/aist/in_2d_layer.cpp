#include <cmath>
#include "in_2d_layer.hpp"

const uint32_t code[] = {
#include "aist/in_2d.comp.h"
};

vkBasalt::aist::In2D::Specialization::Specialization(VkExtent2D extent2D, uint32_t spatialDivisor, uint32_t channels)
    : width(extent2D.width), height(extent2D.height), spatialDivisor(spatialDivisor), channels(channels) {}

vkBasalt::aist::In2D::In2D(
        LogicalDevice *pDevice,
        VkExtent2D extent2D,
        uint32_t chainCount,
        uint32_t spatialDivisor,
        uint32_t channels
) : Layer(pDevice, extent2D, chainCount), specialization(extent2D, spatialDivisor, channels) {
    imageSizeProportion = spatialDivisor;
    depth = 1;
}

void vkBasalt::aist::In2D::createLayout(DsCounterHolder *counters) {
    Layer::createLayout(false);
    counters->uniforms++;
    counters->intermediates += chainCount * 2;
}

void vkBasalt::aist::In2D::createPipelineLayout() {
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

void vkBasalt::aist::In2D::createPipeline() {
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

    createPipelineLayout();

    uint32_t constIdx = 0;
    VkSpecializationMapEntry specEntries[]{
            {.constantID = constIdx++, .offset=offsetof(Specialization, width), .size=sizeof(Specialization::width)},
            {.constantID = constIdx++, .offset=offsetof(Specialization, height), .size=sizeof(Specialization::height)},
            {.constantID = constIdx++, .offset=offsetof(Specialization, spatialDivisor), .size=sizeof(Specialization::spatialDivisor)},
            {.constantID = constIdx++, .offset=offsetof(Specialization, channels), .size=sizeof(Specialization::channels)},
    };
    VkSpecializationInfo specInfo{
            .mapEntryCount = constIdx, .pMapEntries = specEntries,
            .dataSize = sizeof(specialization), .pData = &specialization,
    };
    VkComputePipelineCreateInfo computePipelineCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .stage = {
                    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = nullptr,
                    .flags = 0,
                    .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                    .module = computeModule, .pName = "main",
                    .pSpecializationInfo = &specInfo,
            },
            .layout = pipelineLayout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
    };

    result = pLogicalDevice->vkd.CreateComputePipelines(
            pLogicalDevice->device,
            VK_NULL_HANDLE,
            1,
            &computePipelineCreateInfo,
            nullptr,
            &computePipeline
    );
    ASSERT_VULKAN(result)
}

void vkBasalt::aist::In2D::writeSets(DsWriterHolder holder, uint32_t chainIdx) {
    VkWriteDescriptorSet writes[] = {*holder.weights, *holder.intermediate, *holder.intermediate};
    auto pWeightsInfo = const_cast<VkDescriptorBufferInfo *>(holder.weights->pBufferInfo);
    uint32_t weightsSize = alignTo256Bytes(specialization.channels * 2 * 4);
    pWeightsInfo->range = weightsSize;
    writes[0].dstSet = commonDescriptorSet;
    //We're modifying values in place.
    writes[1].dstBinding = 0;
    //But also need a small one to store stats;
    VkDescriptorBufferInfo statsBufferInfo = *holder.intermediate->pBufferInfo;
    statsBufferInfo.range = weightsSize;
    statsBufferInfo.offset = statsBufferInfo.offset == 0
            ? weightsSize
            : (statsBufferInfo.offset - weightsSize);
    writes[2].pBufferInfo = &statsBufferInfo;
    writes[2].dstSet = writes[1].dstSet = perChainDescriptorSets[chainIdx];
    Layer::writeSets(std::size(writes), writes);
}

void vkBasalt::aist::In2D::appendCommands(VkCommandBuffer commandBuffer, uint32_t chainIdx,
                                          VkBufferMemoryBarrier *bufferBarrierDto) {
    uint32_t substage = 0;
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
            1,
            1,
            depth
    );
    pLogicalDevice->vkd.CmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            1, bufferBarrierDto,
            0, nullptr
    );
    substage += 0;
    pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
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
}
