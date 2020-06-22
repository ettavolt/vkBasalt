#include <cmath>

#include "effect_aist.hpp"

#include "command_buffer.hpp"
#include "image_view.hpp"
#include "descriptor_set.hpp"
#include "util.hpp"

#include "aist/fromimage_layer.hpp"
#include "aist/to_image_layer.hpp"

vkBasalt::AistEffect::AistEffect(
        LogicalDevice *pLogicalDevice,
        VkFormat format,
        VkExtent2D imageExtent,
        std::vector<VkImage> inputImages,
        std::vector<VkImage> outputImages,
        Config *pConfig
) :
        pLogicalDevice(pLogicalDevice),
        format(format),
        imageExtent(imageExtent),
        inputImages(inputImages),
        outputImages(outputImages),
        pConfig(pConfig) {
    Logger::debug("in creating AistEffect");

    inputImageViews = createImageViews(pLogicalDevice, format, inputImages);
    Logger::debug("created input ImageViews");
    //Deal with crash during swapchain recreation. Likely, need to wait for active semaphores.
    //relayoutOutputImages();
    outputImageViews = createImageViews(pLogicalDevice, format, outputImages);
    Logger::debug("created output ImageViews");

    uint32_t chainCount = inputImages.size();

    layers.push_back(std::unique_ptr<aist::Layer>(new aist::FromImageLayer(pLogicalDevice, imageExtent, chainCount)));
    layers.push_back(std::unique_ptr<aist::Layer>(new aist::ToImageLayer(pLogicalDevice, imageExtent, chainCount)));

    createLayoutAndDescriptorSets();

    for (const auto &layer : layers) {
        layer->createPipeline();
    }
}

void vkBasalt::AistEffect::relayoutOutputImages() {
    VkCommandBuffer layoutCommandBuffer = allocateCommandBuffer(pLogicalDevice, 1)[0];
    VkCommandBufferBeginInfo beginInfo;
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.pNext = nullptr;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    beginInfo.pInheritanceInfo = nullptr;
    VkResult result = pLogicalDevice->vkd.BeginCommandBuffer(layoutCommandBuffer, &beginInfo);
    ASSERT_VULKAN(result)

    VkImageMemoryBarrier memoryBarrier;
    memoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    memoryBarrier.pNext = nullptr;
    memoryBarrier.srcAccessMask = 0;
    memoryBarrier.dstAccessMask = 0;
    memoryBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // We put a barrier to transfer from PRESENT to GENERAL before compute shader
    // and then from GENERAL to PRESENT for present|transfer after shader.
    memoryBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    memoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    memoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    memoryBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    memoryBarrier.subresourceRange.baseMipLevel = 0;
    memoryBarrier.subresourceRange.levelCount = 1;
    memoryBarrier.subresourceRange.baseArrayLayer = 0;
    memoryBarrier.subresourceRange.layerCount = 1;
    std::vector<VkImageMemoryBarrier> barriers(outputImages.size(), memoryBarrier);
    for (size_t i = 0; i < outputImages.size(); ++i) {
        barriers[i].image = outputImages[i];
    }
    pLogicalDevice->vkd.CmdPipelineBarrier(
            layoutCommandBuffer,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0, 0, nullptr, 0, nullptr, barriers.size(), barriers.data()
    );
    result = pLogicalDevice->vkd.EndCommandBuffer(layoutCommandBuffer);
    ASSERT_VULKAN(result)
    VkSubmitInfo submitInfo;
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = nullptr;
    submitInfo.waitSemaphoreCount = 0;
    submitInfo.pWaitSemaphores = nullptr;
    submitInfo.pWaitDstStageMask = nullptr;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &layoutCommandBuffer;
    // We need to transistion now
    VkFenceCreateInfo fenceInfo;
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.pNext = nullptr;
    fenceInfo.flags = 0;
    VkFence fence;
    result = pLogicalDevice->vkd.CreateFence(pLogicalDevice->device, &fenceInfo, nullptr, &fence);
    ASSERT_VULKAN(result)
    Logger::trace("Fenced relayout");
    result = pLogicalDevice->vkd.QueueSubmit(pLogicalDevice->queue, 1, &submitInfo, fence);
    ASSERT_VULKAN(result)
    Logger::debug("Waiting on initial layout transition of AIST output images.");
    result = pLogicalDevice->vkd.WaitForFences(pLogicalDevice->device, 1, &fence, VK_TRUE, 1'000'000'000);
    ASSERT_VULKAN(result)
    pLogicalDevice->vkd.DestroyFence(pLogicalDevice->device, fence, nullptr);
    pLogicalDevice->vkd.FreeCommandBuffers(pLogicalDevice->device, pLogicalDevice->commandPool, 1,
                                           &layoutCommandBuffer);
    Logger::trace("Relayout complete, extra freed");
}

void vkBasalt::AistEffect::createLayoutAndDescriptorSets() {
    VkResult result;

    std::vector<VkDescriptorSetLayout> commonSetLayouts;
    for (const auto &layer : layers) {
        layer->createLayout();
        commonSetLayouts.push_back(layer->commonDescriptorSetLayout);
    }

    uint32_t chainCount = inputImages.size();
    VkDescriptorPoolSize poolSize;
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSize.descriptorCount = (chainCount + 1) * layers.size();
    descriptorPool = createDescriptorPool(pLogicalDevice, std::vector<VkDescriptorPoolSize>({poolSize}));
    Logger::debug("created descriptorPool");

    for (const auto &layer : layers) {
        std::vector<VkDescriptorSetLayout> layouts(chainCount, layer->perChainDescriptorSetLayout);
        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
                .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .pNext = nullptr,
                .descriptorPool = descriptorPool,
                .descriptorSetCount = chainCount,
        };
        descriptorSetAllocateInfo.pSetLayouts = layouts.data();
        result = pLogicalDevice->vkd.AllocateDescriptorSets(
                pLogicalDevice->device,
                &descriptorSetAllocateInfo,
                layer->perChainDescriptorSets.data()
        );
        ASSERT_VULKAN(result)
        descriptorSetAllocateInfo.descriptorSetCount = 1;
        descriptorSetAllocateInfo.pSetLayouts = &layer->commonDescriptorSetLayout;
        result = pLogicalDevice->vkd.AllocateDescriptorSets(
                pLogicalDevice->device,
                &descriptorSetAllocateInfo,
                &layer->commonDescriptorSet
        );
        ASSERT_VULKAN(result)
    }
    Logger::debug("after allocating descriptor Sets");

    // In & Out
    std::vector<VkDescriptorImageInfo> imageInfos(2);
    std::vector<VkWriteDescriptorSet> writeDescriptorSets(imageInfos.size());
    for (uint32_t i = 0; i < imageInfos.size(); i++) {
        VkDescriptorImageInfo imageInfo;
        imageInfo.sampler = VK_NULL_HANDLE;
        imageInfo.imageView = VK_NULL_HANDLE;
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        imageInfos[i] = imageInfo;
        VkWriteDescriptorSet writeDescriptorSet;
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.pNext = nullptr;
        writeDescriptorSet.dstSet = VK_NULL_HANDLE;
        writeDescriptorSet.dstArrayElement = 0;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writeDescriptorSet.pImageInfo = &(imageInfos[i]);
        writeDescriptorSet.pBufferInfo = nullptr;
        writeDescriptorSet.pTexelBufferView = nullptr;
        writeDescriptorSets[i] = writeDescriptorSet;
    }

    Logger::debug("before writing descriptor Sets");
    for (uint32_t chainIdx = 0; chainIdx < chainCount; chainIdx++) {
        imageInfos[0].imageView = inputImageViews[chainIdx];
        imageInfos[1].imageView = outputImageViews[chainIdx];
        for (const auto &layer : layers) {
            layer->writeSets(&(writeDescriptorSets[0]), &(writeDescriptorSets[1]), chainIdx);
        }
    }
    Logger::debug("after writing descriptor Sets");
}

void vkBasalt::AistEffect::applyEffect(uint32_t imageIndex, VkCommandBuffer commandBuffer) {
    Logger::debug("applying AistEffect to cb " + convertToString(commandBuffer));
    // After shader has run modify layout of output image again to support present|transfer.
    VkImageMemoryBarrier outputAfterShaderBarrier;
    outputAfterShaderBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    outputAfterShaderBarrier.pNext = nullptr;
    outputAfterShaderBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    outputAfterShaderBarrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    outputAfterShaderBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    outputAfterShaderBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    outputAfterShaderBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputAfterShaderBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    outputAfterShaderBarrier.image = outputImages[imageIndex];
    outputAfterShaderBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    outputAfterShaderBarrier.subresourceRange.baseMipLevel = 0;
    outputAfterShaderBarrier.subresourceRange.levelCount = 1;
    outputAfterShaderBarrier.subresourceRange.baseArrayLayer = 0;
    outputAfterShaderBarrier.subresourceRange.layerCount = 1;

    std::vector<VkImageMemoryBarrier> beforeShaderBarriers(2, outputAfterShaderBarrier);

    // Need to wait for previous effect and also convert layout…
    beforeShaderBarriers[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    beforeShaderBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    beforeShaderBarriers[0].oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    beforeShaderBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    beforeShaderBarriers[1].image = inputImages[imageIndex];

    // …and convert layout of output image as it is unmodified after previous execution of this (same!) buffer.
    beforeShaderBarriers[1].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    beforeShaderBarriers[1].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    beforeShaderBarriers[1].oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    beforeShaderBarriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;

    pLogicalDevice->vkd.CmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            beforeShaderBarriers.size(), beforeShaderBarriers.data()
    );
    Logger::debug("after the input pipeline barriers");

    for (const auto &layer : layers) {
        pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layer->computePipeline);
        Logger::debug("after bind pipeline");

        VkDescriptorSet sets[]{
                layer->commonDescriptorSet,
                layer->perChainDescriptorSets[imageIndex],
        };
        pLogicalDevice->vkd.CmdBindDescriptorSets(
                commandBuffer,
                VK_PIPELINE_BIND_POINT_COMPUTE,
                layer->pipelineLayout, 0,
                std::size(sets), sets,
                0, nullptr
        );
        Logger::debug("after binding image storage");

        pLogicalDevice->vkd.CmdDispatch(
                commandBuffer,
                (uint32_t) std::ceil(imageExtent.width / 16.0),
                (uint32_t) std::ceil(imageExtent.height / 16.0),
                1
        );
        Logger::debug("after dispatch");
    }

    pLogicalDevice->vkd.CmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &outputAfterShaderBarrier
    );

    Logger::debug("after the output pipeline barrier");
}

vkBasalt::AistEffect::~AistEffect() {
    Logger::debug("destroying AistEffect " + convertToString(this));
    layers.clear();
    pLogicalDevice->vkd.DestroyDescriptorPool(pLogicalDevice->device, descriptorPool, nullptr);

    for (unsigned int i = 0; i < inputImageViews.size(); i++) {
        pLogicalDevice->vkd.DestroyImageView(pLogicalDevice->device, inputImageViews[i], nullptr);
        pLogicalDevice->vkd.DestroyImageView(pLogicalDevice->device, outputImageViews[i], nullptr);
    }
    Logger::debug("after DestroyImageView");
}

