#include <CLucene.h>
#include "effect_aist.hpp"

#include "command_buffer.hpp"
#include "image_view.hpp"
#include "descriptor_set.hpp"
#include "util.hpp"

namespace vkBasalt
{
    const uint32_t AistEffect::computeCode[] = {
#include "aist.comp.h"
    };

    AistEffect::AistEffect(LogicalDevice*       pLogicalDevice,
                           VkFormat             format,
                           VkExtent2D           imageExtent,
                           std::vector<VkImage> inputImages,
                           std::vector<VkImage> outputImages,
                           Config*              pConfig)
    {
        Logger::debug("in creating AistEffect");

        this->pLogicalDevice = pLogicalDevice;
        this->format         = format;
        this->imageExtent    = imageExtent;
        this->inputImages    = inputImages;
        this->outputImages   = outputImages;
        this->pConfig        = pConfig;

        inputImageViews = createImageViews(pLogicalDevice, format, inputImages);
        Logger::debug("created input ImageViews");
        relayoutOutputImages();
        outputImageViews = createImageViews(pLogicalDevice, format, outputImages);
        Logger::debug("created output ImageViews");

        createLayoutAndDescriptorSets();

        createPipeline();
    }

    void AistEffect::relayoutOutputImages()
    {
        VkCommandBuffer layoutCommandBuffer = allocateCommandBuffer(pLogicalDevice, 1)[0];
        VkCommandBufferBeginInfo beginInfo;
        beginInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.pNext            = nullptr;
        beginInfo.flags            = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        beginInfo.pInheritanceInfo = nullptr;
        VkResult result = pLogicalDevice->vkd.BeginCommandBuffer(layoutCommandBuffer, &beginInfo);
        ASSERT_VULKAN(result)

        VkImageMemoryBarrier memoryBarrier;
        memoryBarrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        memoryBarrier.pNext               = nullptr;
        memoryBarrier.srcAccessMask       = 0;
        memoryBarrier.dstAccessMask       = 0;
        memoryBarrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        // We put a barrier to transfer from PRESENT to GENERAL before compute shader
        // and then from GENERAL to PRESENT for present|transfer after shader.
        memoryBarrier.newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        memoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memoryBarrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        memoryBarrier.subresourceRange.baseMipLevel   = 0;
        memoryBarrier.subresourceRange.levelCount     = 1;
        memoryBarrier.subresourceRange.baseArrayLayer = 0;
        memoryBarrier.subresourceRange.layerCount     = 1;
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
        submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.pNext              = nullptr;
        submitInfo.waitSemaphoreCount = 0;
        submitInfo.pWaitSemaphores    = nullptr;
        submitInfo.pWaitDstStageMask  = nullptr;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers    = &layoutCommandBuffer;
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
        pLogicalDevice->vkd.FreeCommandBuffers(pLogicalDevice->device, pLogicalDevice->commandPool, 1, &layoutCommandBuffer);
        Logger::trace("Relayout complete, extra freed");
    }

    void AistEffect::createLayoutAndDescriptorSets()
    {
        uint32_t imageCount = inputImages.size();

        // In & Out
        std::vector<VkDescriptorSetLayoutBinding> bindings(2);
        std::vector<VkDescriptorImageInfo> imageInfos(bindings.size());
        std::vector<VkWriteDescriptorSet> writeDescriptorSets(bindings.size());
        for (uint32_t i = 0; i < 2; i++) {
            VkDescriptorSetLayoutBinding descriptorSetLayoutBinding;
            uint32_t bindingIndex = 2 + i;
            descriptorSetLayoutBinding.binding = bindingIndex;
            descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBinding.descriptorCount = 1;
            descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            descriptorSetLayoutBinding.pImmutableSamplers = nullptr;
            bindings[i] = descriptorSetLayoutBinding;
            VkDescriptorImageInfo imageInfo;
            imageInfo.sampler     = VK_NULL_HANDLE;
            imageInfo.imageView   = VK_NULL_HANDLE;
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            imageInfos[i] = imageInfo;
            VkWriteDescriptorSet writeDescriptorSet;
            writeDescriptorSet.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.pNext            = nullptr;
            writeDescriptorSet.dstSet           = VK_NULL_HANDLE;
            writeDescriptorSet.dstBinding       = bindingIndex;
            writeDescriptorSet.dstArrayElement  = 0;
            writeDescriptorSet.descriptorCount  = 1;
            writeDescriptorSet.descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writeDescriptorSet.pImageInfo       = &(imageInfos[i]);
            writeDescriptorSet.pBufferInfo      = nullptr;
            writeDescriptorSet.pTexelBufferView = nullptr;
            writeDescriptorSets[i] = writeDescriptorSet;
        }

        VkDescriptorSetLayoutCreateInfo descriptorSetCreateInfo;
        descriptorSetCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetCreateInfo.pNext        = nullptr;
        descriptorSetCreateInfo.flags        = 0;
        descriptorSetCreateInfo.bindingCount = bindings.size();
        descriptorSetCreateInfo.pBindings    = bindings.data();
        VkResult result = pLogicalDevice->vkd.CreateDescriptorSetLayout(
                pLogicalDevice->device,
                &descriptorSetCreateInfo,
                nullptr,
                &descriptorSetLayout
        );
        ASSERT_VULKAN(result)
        Logger::debug("created descriptorSetLayout");

        VkDescriptorPoolSize poolSize;
        poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        poolSize.descriptorCount = imageCount;
        descriptorPool = createDescriptorPool(pLogicalDevice, std::vector<VkDescriptorPoolSize>({poolSize}));
        Logger::debug("created descriptorPool");

        imageDescriptorSets.resize(imageCount);
        std::vector<VkDescriptorSetLayout> layouts(imageCount, descriptorSetLayout);
        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
        descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.pNext              = nullptr;
        descriptorSetAllocateInfo.descriptorPool     = descriptorPool;
        descriptorSetAllocateInfo.descriptorSetCount = imageCount;
        descriptorSetAllocateInfo.pSetLayouts        = layouts.data();
        Logger::debug("before allocating descriptor Sets");
        result = pLogicalDevice->vkd.AllocateDescriptorSets(pLogicalDevice->device, &descriptorSetAllocateInfo, imageDescriptorSets.data());
        ASSERT_VULKAN(result)
        Logger::debug("after allocating descriptor Sets");

        for (unsigned int i = 0; i < imageCount; i++)
        {
            imageInfos[0].imageView = inputImageViews[i];
            imageInfos[1].imageView = outputImageViews[i];
            writeDescriptorSets[0].dstSet = imageDescriptorSets[i];
            writeDescriptorSets[1].dstSet = imageDescriptorSets[i];
            Logger::debug("before writing descriptor Sets");
            pLogicalDevice->vkd.UpdateDescriptorSets(
                    pLogicalDevice->device,
                    writeDescriptorSets.size(),
                    writeDescriptorSets.data(),
                    0,
                    nullptr
            );
        }
    }

    void AistEffect::createPipeline() {
        VkPushConstantRange pushConstantRange;
        pushConstantRange.size = sizeof(VkExtent2D);
        pushConstantRange.offset = 0;
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo;
        pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutCreateInfo.pNext = nullptr;
        pipelineLayoutCreateInfo.flags = 0;
        pipelineLayoutCreateInfo.setLayoutCount = 1;
        pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
        pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

        VkResult result = pLogicalDevice->vkd.CreatePipelineLayout(
                pLogicalDevice->device,
                &pipelineLayoutCreateInfo,
                nullptr,
                &pipelineLayout
        );
        ASSERT_VULKAN(result)

        VkShaderModuleCreateInfo shaderCreateInfo;
        shaderCreateInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderCreateInfo.pNext    = nullptr;
        shaderCreateInfo.flags    = 0;
        shaderCreateInfo.codeSize = sizeof(computeCode);
        shaderCreateInfo.pCode    = computeCode;
        result = pLogicalDevice->vkd.CreateShaderModule(pLogicalDevice->device, &shaderCreateInfo, nullptr, &computeModule);
        ASSERT_VULKAN(result);
        VkPipelineShaderStageCreateInfo shaderStageCreateInfo;
        shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageCreateInfo.pNext = nullptr;
        shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageCreateInfo.flags = 0;
        shaderStageCreateInfo.module = computeModule;
        shaderStageCreateInfo.pName = "main";
        shaderStageCreateInfo.pSpecializationInfo = nullptr;

        VkComputePipelineCreateInfo computePipelineCreateInfo;
        computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        computePipelineCreateInfo.pNext = nullptr;
        computePipelineCreateInfo.flags = 0;
        computePipelineCreateInfo.layout = pipelineLayout;
        computePipelineCreateInfo.stage = shaderStageCreateInfo;
        computePipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
        computePipelineCreateInfo.basePipelineIndex = -1;

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

    void AistEffect::applyEffect(uint32_t imageIndex, VkCommandBuffer commandBuffer)
    {
        Logger::debug("applying AistEffect to cb " + convertToString(commandBuffer));
        // After shader has run modify layout of output image again to support present|transfer.
        VkImageMemoryBarrier outputAfterShaderBarrier;
        outputAfterShaderBarrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        outputAfterShaderBarrier.pNext               = nullptr;
        outputAfterShaderBarrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
        outputAfterShaderBarrier.dstAccessMask       = VK_ACCESS_MEMORY_READ_BIT;
        outputAfterShaderBarrier.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
        outputAfterShaderBarrier.newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        outputAfterShaderBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        outputAfterShaderBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        outputAfterShaderBarrier.image               = outputImages[imageIndex];
        outputAfterShaderBarrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        outputAfterShaderBarrier.subresourceRange.baseMipLevel   = 0;
        outputAfterShaderBarrier.subresourceRange.levelCount     = 1;
        outputAfterShaderBarrier.subresourceRange.baseArrayLayer = 0;
        outputAfterShaderBarrier.subresourceRange.layerCount     = 1;

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

        pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        Logger::debug("after bind pipeline");

        pLogicalDevice->vkd.CmdBindDescriptorSets(
            commandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipelineLayout, 0,
            1, &(imageDescriptorSets[imageIndex]),
            0, nullptr
        );
        Logger::debug("after binding image storage");

        pLogicalDevice->vkd.CmdPushConstants(
            commandBuffer,
            pipelineLayout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0,
            sizeof(VkExtent2D),
            &imageExtent
        );
        Logger::debug("after binding image storage");

        pLogicalDevice->vkd.CmdDispatch(commandBuffer, 1, 1, 1);
        Logger::debug("after dispatch");

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

    AistEffect::~AistEffect()
    {
        Logger::debug("destroying AistEffect " + convertToString(this));
        pLogicalDevice->vkd.DestroyPipeline(pLogicalDevice->device, computePipeline, nullptr);
        pLogicalDevice->vkd.DestroyShaderModule(pLogicalDevice->device, computeModule, nullptr);
        pLogicalDevice->vkd.DestroyPipelineLayout(pLogicalDevice->device, pipelineLayout, nullptr);
        pLogicalDevice->vkd.DestroyDescriptorSetLayout(pLogicalDevice->device, descriptorSetLayout, nullptr);
        pLogicalDevice->vkd.DestroyDescriptorPool(pLogicalDevice->device, descriptorPool, nullptr);

        for (unsigned int i = 0; i < inputImageViews.size(); i++)
        {
            pLogicalDevice->vkd.DestroyImageView(pLogicalDevice->device, inputImageViews[i], nullptr);
            pLogicalDevice->vkd.DestroyImageView(pLogicalDevice->device, outputImageViews[i], nullptr);
        }
        Logger::debug("after DestroyImageView");
    }
} // namespace vkBasalt
