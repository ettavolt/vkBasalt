#include "effect_aist.hpp"

#include "image_view.hpp"
#include "descriptor_set.hpp"
#include "graphics_pipeline.hpp"
#include "shader.hpp"
#include "util.hpp"

#include "shader_sources.hpp"

namespace vkBasalt
{
    AistEffect::AistEffect(LogicalDevice*       pLogicalDevice,
                            VkFormat             format,
                            std::vector<VkImage> inputImages,
                            std::vector<VkImage> outputImages,
                            Config*              pConfig)
    {
        Logger::debug("in creating AistEffect");

        this->pLogicalDevice = pLogicalDevice;
        this->format         = format;
        this->inputImages    = inputImages;
        this->outputImages   = outputImages;
        this->pConfig        = pConfig;
        computeCode = aist_comp;

        inputImageViews = createImageViews(pLogicalDevice, format, inputImages);
        Logger::debug("created input ImageViews");
        outputImageViews = createImageViews(pLogicalDevice, format, outputImages);
        Logger::debug("created output ImageViews");

        createLayoutAndDescriptorSets();

        pipelineLayout = createGraphicsPipelineLayout(
                pLogicalDevice,
                std::vector<VkDescriptorSetLayout>({descriptorSetLayout})
        );

        createShaderModule(pLogicalDevice, computeCode, &computeModule);
        VkPipelineShaderStageCreateInfo shaderStageCreateInfo;
        shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageCreateInfo.pNext = nullptr;
        shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
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

        VkResult result = pLogicalDevice->vkd.CreateComputePipelines(
                pLogicalDevice->device,
                VK_NULL_HANDLE,
                1,
                &computePipelineCreateInfo,
                nullptr,
                &computePipeline
        );
        ASSERT_VULKAN(result)
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
            descriptorSetLayoutBinding.binding = i;
            descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorSetLayoutBinding.descriptorCount = 1;
            descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            descriptorSetLayoutBinding.pImmutableSamplers = nullptr;
            bindings.push_back(descriptorSetLayoutBinding);
            VkDescriptorImageInfo imageInfo = {};
            imageInfo.sampler     = VK_NULL_HANDLE;
            imageInfo.imageView   = VK_NULL_HANDLE;
            imageInfo.imageLayout = i == 0
                    ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    : VK_IMAGE_LAYOUT_GENERAL;
            imageInfos.push_back(imageInfo);
            VkWriteDescriptorSet writeDescriptorSet = {};
            writeDescriptorSet.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writeDescriptorSet.pNext            = nullptr;
            writeDescriptorSet.dstSet           = VK_NULL_HANDLE;
            writeDescriptorSet.dstBinding       = i;
            writeDescriptorSet.dstArrayElement  = 0;
            writeDescriptorSet.descriptorCount  = 1;
            writeDescriptorSet.descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writeDescriptorSet.pImageInfo       = &imageInfo;
            writeDescriptorSet.pBufferInfo      = nullptr;
            writeDescriptorSet.pTexelBufferView = nullptr;
            writeDescriptorSets.push_back(writeDescriptorSet);
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
        VkDescriptorSetAllocateInfo descriptorSetAllocateInfo;
        descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptorSetAllocateInfo.pNext              = nullptr;
        descriptorSetAllocateInfo.descriptorPool     = descriptorPool;
        descriptorSetAllocateInfo.descriptorSetCount = imageCount;
        descriptorSetAllocateInfo.pSetLayouts        = &descriptorSetLayout;
        Logger::debug("before allocating descriptor Sets");
        result = pLogicalDevice->vkd.AllocateDescriptorSets(pLogicalDevice->device, &descriptorSetAllocateInfo, imageDescriptorSets.data());
        ASSERT_VULKAN(result);
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

    void AistEffect::applyEffect(uint32_t imageIndex, VkCommandBuffer commandBuffer)
    {
        Logger::debug("applying AistEffect to cb " + convertToString(commandBuffer));
        // Used to make the Image accessable by the shader
        VkImageMemoryBarrier memoryBarrier;
        memoryBarrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        memoryBarrier.pNext               = nullptr;
        memoryBarrier.srcAccessMask       = VK_ACCESS_MEMORY_WRITE_BIT;
        memoryBarrier.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        memoryBarrier.oldLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        memoryBarrier.newLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        memoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        memoryBarrier.image               = inputImages[imageIndex];

        memoryBarrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        memoryBarrier.subresourceRange.baseMipLevel   = 0;
        memoryBarrier.subresourceRange.levelCount     = 1;
        memoryBarrier.subresourceRange.baseArrayLayer = 0;
        memoryBarrier.subresourceRange.layerCount     = 1;

        // Reverses the first Barrier
        VkImageMemoryBarrier secondBarrier;
        secondBarrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        secondBarrier.pNext               = nullptr;
        secondBarrier.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT;
        secondBarrier.dstAccessMask       = 0;//VK_ACCESS_MEMORY_READ_BIT;
        secondBarrier.oldLayout           = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        secondBarrier.newLayout           = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        secondBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        secondBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        secondBarrier.image               = inputImages[imageIndex];

        secondBarrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        secondBarrier.subresourceRange.baseMipLevel   = 0;
        secondBarrier.subresourceRange.levelCount     = 1;
        secondBarrier.subresourceRange.baseArrayLayer = 0;
        secondBarrier.subresourceRange.layerCount     = 1;

        pLogicalDevice->vkd.CmdPipelineBarrier(
            commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &memoryBarrier);
        Logger::debug("after the first pipeline barrier");

        pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        Logger::debug("after bind pipeline");

        pLogicalDevice->vkd.CmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &(imageDescriptorSets[imageIndex]), 0, nullptr);
        Logger::debug("after binding image storage");

        pLogicalDevice->vkd.CmdDispatch(commandBuffer, 1, 1, 1);
        Logger::debug("after dispatch");

        pLogicalDevice->vkd.CmdPipelineBarrier(commandBuffer,
                                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                               0,
                                               0,
                                               nullptr,
                                               0,
                                               nullptr,
                                               1,
                                               &secondBarrier);
        Logger::debug("after the second pipeline barrier");
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
