#include <cmath>
#include "to_image_layer.hpp"

const uint32_t code[] = {
#include "aist/to_image.comp.h"
};

vkBasalt::aist::ToImageLayer::ToImageLayer(LogicalDevice *pDevice, VkExtent2D extent2D, uint32_t chainCount)
        : Layer(pDevice, extent2D, chainCount) {}

void vkBasalt::aist::ToImageLayer::createLayout(DsCounterHolder *counters) {
    uint32_t bindingIndex = 0;
    VkDescriptorSetLayoutBinding storageBindings[]{
            {
                    .binding = bindingIndex++,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    .pImmutableSamplers = nullptr,
            },
            {
                    .binding = bindingIndex++,
                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    .pImmutableSamplers = nullptr,
            }
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = bindingIndex,
            .pBindings    = storageBindings,
    };
    VkResult result = pLogicalDevice->vkd.CreateDescriptorSetLayout(
            pLogicalDevice->device,
            &descriptorSetCreateInfo,
            nullptr,
            &perChainDescriptorSetLayout
    );
    ASSERT_VULKAN(result)
    counters->images += chainCount;
    counters->intermediates += chainCount;
}

void vkBasalt::aist::ToImageLayer::createPipelineLayout() {
    VkDescriptorSetLayout setLayouts[]{
            perChainDescriptorSetLayout,
    };
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .setLayoutCount = std::size(setLayouts),
            .pSetLayouts = setLayouts,
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr,
    };
    VkResult result = pLogicalDevice->vkd.CreatePipelineLayout(
            pLogicalDevice->device,
            &pipelineLayoutCreateInfo,
            nullptr,
            &pipelineLayout
    );
    ASSERT_VULKAN(result)
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
    VkWriteDescriptorSet writes[] = {*holder.outImage, *holder.intermediate};
    writes[0].dstSet = writes[1].dstSet = perChainDescriptorSets[chainIdx];
    Layer::writeSets(std::size(writes), writes);
}

void vkBasalt::aist::ToImageLayer::appendCommands(VkCommandBuffer commandBuffer, uint32_t chainIdx,
                                                  VkBufferMemoryBarrier *bufferBarrierDto) {
    pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
    VkDescriptorSet sets[]{
            perChainDescriptorSets[chainIdx],
    };
    pLogicalDevice->vkd.CmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout, 0,
            std::size(sets), sets,
            0, nullptr
    );
    pLogicalDevice->vkd.CmdDispatch(
            commandBuffer,
            (uint32_t) std::ceil(imageExtent.width / imageSizeProportion),
            (uint32_t) std::ceil(imageExtent.height / imageSizeProportion),
            depth
    );
}
