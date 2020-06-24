#include <cmath>
#include "nn_layer.h"

vkBasalt::aist::Layer::Layer(LogicalDevice *pLogicalDevice, VkExtent2D imageExtent, uint32_t chainCount) :
        pLogicalDevice(pLogicalDevice), imageExtent(imageExtent), chainCount(chainCount) {
    perChainDescriptorSets.resize(chainCount);
}

void vkBasalt::aist::Layer::createPipeline() {
    VkDescriptorSetLayout setLayouts[]{
            commonDescriptorSetLayout,
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

    uint32_t constIdx = 0;
    VkSpecializationMapEntry specEntries[]{
            {.constantID = constIdx++, .offset=offsetof(VkExtent2D, width), .size=sizeof(VkExtent2D::width)},
            {.constantID = constIdx++, .offset=offsetof(VkExtent2D, height), .size=sizeof(VkExtent2D::height)},
    };
    VkSpecializationInfo specInfo{
            .mapEntryCount = constIdx, .pMapEntries = specEntries,
            .dataSize = sizeof(imageExtent), .pData = &imageExtent,
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

vkBasalt::aist::Layer::~Layer() {
    if (computePipeline != VK_NULL_HANDLE) {
        pLogicalDevice->vkd.DestroyPipeline(pLogicalDevice->device, computePipeline, nullptr);
    }
    if (computeModule != VK_NULL_HANDLE) {
        pLogicalDevice->vkd.DestroyShaderModule(pLogicalDevice->device, computeModule, nullptr);
    }
    if (pipelineLayout != VK_NULL_HANDLE) {
        pLogicalDevice->vkd.DestroyPipelineLayout(pLogicalDevice->device, pipelineLayout, nullptr);
    }
    if (perChainDescriptorSetLayout != nullptr) {
        pLogicalDevice->vkd.DestroyDescriptorSetLayout(
                pLogicalDevice->device,
                perChainDescriptorSetLayout,
                nullptr
        );
    }
    if (commonDescriptorSetLayout != nullptr) {
        pLogicalDevice->vkd.DestroyDescriptorSetLayout(
                pLogicalDevice->device,
                commonDescriptorSetLayout,
                nullptr
        );
    }
}

void vkBasalt::aist::Layer::createLayout(bool tapsIntoImage) {
    uint32_t bindingIndex = 0;
    VkDescriptorSetLayoutBinding weightsBindings[]{
            {
                    .binding = bindingIndex++,
                    .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    .descriptorCount = 1,
                    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                    .pImmutableSamplers = nullptr,
            },
    };
    VkDescriptorSetLayoutCreateInfo descriptorSetCreateInfo = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = bindingIndex,
            .pBindings    = weightsBindings,
    };

    VkResult result = pLogicalDevice->vkd.CreateDescriptorSetLayout(
            pLogicalDevice->device,
            &descriptorSetCreateInfo,
            nullptr,
            &commonDescriptorSetLayout
    );
    ASSERT_VULKAN(result)

    bindingIndex = 0;
    VkDescriptorSetLayoutBinding storageBindings[]{
            {
                    .binding = bindingIndex++,
                    .descriptorType = tapsIntoImage
                                      ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                      : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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

    descriptorSetCreateInfo.bindingCount = bindingIndex;
    descriptorSetCreateInfo.pBindings = storageBindings;
    result = pLogicalDevice->vkd.CreateDescriptorSetLayout(
            pLogicalDevice->device,
            &descriptorSetCreateInfo,
            nullptr,
            &perChainDescriptorSetLayout
    );
    ASSERT_VULKAN(result)
}

void vkBasalt::aist::Layer::writeSets(uint32_t count, VkWriteDescriptorSet *writes) {
    pLogicalDevice->vkd.UpdateDescriptorSets(
            pLogicalDevice->device,
            count,
            writes,
            0,
            nullptr
    );
}

void vkBasalt::aist::Layer::dispatchPipeline(VkCommandBuffer commandBuffer, uint32_t chainIdx) {
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
    pLogicalDevice->vkd.CmdDispatch(
            commandBuffer,
            (uint32_t) std::ceil(imageExtent.width / imageSizeProportion),
            (uint32_t) std::ceil(imageExtent.height / imageSizeProportion),
            1
    );
}

void vkBasalt::aist::Layer::appendCommands(VkCommandBuffer commandBuffer, uint32_t chainIdx) {
    dispatchPipeline(commandBuffer, chainIdx);
}
