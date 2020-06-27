#include <cmath>
#include <numeric>

#include "effect_aist.hpp"

#include "buffer.hpp"
#include "command_buffer.hpp"
#include "image_view.hpp"
#include "descriptor_set.hpp"
#include "util.hpp"

#include "aist/fromimage_layer.hpp"
#include "aist/up_conv_32_3_layer.hpp"
#include "aist/to_image_layer.hpp"
#include "memory.hpp"

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

    allocateBuffers();
    Logger::debug("allocated buffers");

    inputImageViews = createImageViews(pLogicalDevice, format, inputImages);
    Logger::debug("created input ImageViews");
    //Deal with crash during swapchain recreation. Likely, need to wait for active semaphores.
    //relayoutOutputImages();
    outputImageViews = createImageViews(pLogicalDevice, format, outputImages);
    Logger::debug("created output ImageViews");

    uint32_t chainCount = inputImages.size();

    layers.push_back(std::unique_ptr<aist::Layer>(new aist::FromImageLayer(pLogicalDevice, imageExtent, chainCount)));
    layers.push_back(std::unique_ptr<aist::Layer>(new aist::UpConv32t3(pLogicalDevice, imageExtent, chainCount)));
    layers.push_back(std::unique_ptr<aist::Layer>(new aist::ToImageLayer(pLogicalDevice, imageExtent, chainCount)));

    createLayoutAndDescriptorSets();

    for (const auto &layer : layers) {
        layer->createPipeline();
    }
}

void vkBasalt::AistEffect::allocateBuffers() {
    auto weightsFileName = pConfig->getOption<std::string>("aistWeigthsFile");
    std::ifstream file(weightsFileName, std::ios::in|std::ios::binary|std::ios::ate);
    if (!file.is_open()) {
        Logger::err("Can't open aistWeigthsFile!");
    }
    auto fileSize = file.tellg();
    // Should be:
    // ((3 × 9 + 3 × 32) + (32 × 9 + 32 × 64)) × 2 + (64 × 9 + 64 × 64) × 2 × 15
    // 4 - 32-bit float.
    VkDeviceSize weightsSize = fileSize;
    char* weightsHostBuf = new char [fileSize];
    file.seekg(0, std::ios::beg);
    file.read(weightsHostBuf, fileSize);
    file.close();
    Logger::debug("Loaded aistWeigthsFile");
    VkBufferCreateInfo bufferInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = weightsSize,
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkResult result = pLogicalDevice->vkd.CreateBuffer(pLogicalDevice->device, &bufferInfo, nullptr, &weights);
    ASSERT_VULKAN(result)
    // (width / 2) × (height / 2) × 32 + (width / 4) × (height / 4) × 64
    //        spatial reduction|    |channels     |more SR    channels|
    // w x h x 12 of 32-bit floats.
    VkDeviceSize intermediateSize = imageExtent.width * imageExtent.height * 12 * 4;
    bufferInfo.size = intermediateSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    intermediates.resize(inputImages.size());
    for (auto & intermediate : intermediates)
    {
        result = pLogicalDevice->vkd.CreateBuffer(pLogicalDevice->device, &bufferInfo, nullptr, &intermediate);
        ASSERT_VULKAN(result)
    }
    VkMemoryRequirements weightsMemReqs;
    pLogicalDevice->vkd.GetBufferMemoryRequirements(pLogicalDevice->device, weights, &weightsMemReqs);
    VkMemoryRequirements intermediateMemReqs;
    pLogicalDevice->vkd.GetBufferMemoryRequirements(pLogicalDevice->device, intermediates[0], &intermediateMemReqs);
    Logger::debug("AIST: created buffer handles and got mem reqs.");

    const VkDeviceSize alignment = std::lcm(weightsMemReqs.alignment, intermediateMemReqs.alignment);
    VkDeviceSize memOffset = intermediateSize;
    VkDeviceSize overAlignment = memOffset % alignment;
    if (overAlignment > 0) memOffset += alignment - overAlignment;
    memOffset *= intermediates.size();
    memOffset += weightsSize;
    overAlignment = weightsSize % alignment;
    if (overAlignment > 0) memOffset += alignment - overAlignment;

    VkMemoryAllocateInfo allocInfo = {
        .sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize  = memOffset,
        .memoryTypeIndex = findMemoryTypeIndex(
                pLogicalDevice,
                weightsMemReqs.memoryTypeBits | intermediateMemReqs.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        ),
    };
    result = pLogicalDevice->vkd.AllocateMemory(pLogicalDevice->device, &allocInfo, nullptr, &bufferMemory);
    ASSERT_VULKAN(result)

    memOffset = 0;
    result = pLogicalDevice->vkd.BindBufferMemory(pLogicalDevice->device, weights, bufferMemory, memOffset);
    ASSERT_VULKAN(result)
    memOffset = weightsSize;
    overAlignment = weightsSize % alignment;
    if (overAlignment > 0) memOffset += alignment - overAlignment;
    overAlignment = intermediateSize % alignment;
    if (overAlignment > 0) overAlignment = alignment - overAlignment;
    for (const auto &intermediate : intermediates) {
        result = pLogicalDevice->vkd.BindBufferMemory(pLogicalDevice->device, intermediate, bufferMemory, memOffset);
        ASSERT_VULKAN(result)
        memOffset += intermediateSize + overAlignment;
    }
    Logger::debug("AIST: bound mem to buffers.");

    VkBuffer       stagingBuffer;
    VkDeviceMemory stagingMemory;
    createBuffer(pLogicalDevice,
                 weightsSize,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 stagingBuffer,
                 stagingMemory);
    void* weightsCoherentBuf;
    result = pLogicalDevice->vkd.MapMemory(pLogicalDevice->device, stagingMemory, 0, weightsSize, 0, &weightsCoherentBuf);
    ASSERT_VULKAN(result)
    std::memcpy(weightsCoherentBuf, weightsHostBuf, weightsSize);
    pLogicalDevice->vkd.UnmapMemory(pLogicalDevice->device, stagingMemory);
    Logger::debug("AIST: staged weights.");

    VkCommandBufferAllocateInfo cbAllocInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = pLogicalDevice->commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
    };
    VkCommandBuffer commandBuffer;
    pLogicalDevice->vkd.AllocateCommandBuffers(pLogicalDevice->device, &cbAllocInfo, &commandBuffer);
    // initialize dispatch table for commandBuffer since it is a dispatchable object
    initializeDispatchTable(commandBuffer, pLogicalDevice->device);

    VkCommandBufferBeginInfo beginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
    };
    pLogicalDevice->vkd.BeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy region{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = weightsSize,
    };
    pLogicalDevice->vkd.CmdCopyBuffer(commandBuffer, stagingBuffer, weights, 1, &region);

    pLogicalDevice->vkd.EndCommandBuffer(commandBuffer);
    VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
    };
    Logger::debug("AIST: transferring weights.");
    pLogicalDevice->vkd.QueueSubmit(pLogicalDevice->queue, 1, &submitInfo, VK_NULL_HANDLE);
    pLogicalDevice->vkd.QueueWaitIdle(pLogicalDevice->queue);

    pLogicalDevice->vkd.FreeCommandBuffers(pLogicalDevice->device, pLogicalDevice->commandPool, 1, &commandBuffer);
    pLogicalDevice->vkd.FreeMemory(pLogicalDevice->device, stagingMemory, nullptr);
    pLogicalDevice->vkd.DestroyBuffer(pLogicalDevice->device, stagingBuffer, nullptr);
    Logger::debug("AIST: transferred weights.");
}

void vkBasalt::AistEffect::relayoutOutputImages() {
    auto layoutCommandBuffer = allocateCommandBuffer(pLogicalDevice, 1)[0];
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
    aist::DsCounterHolder counters{
            .images = 0,
            .uniforms = 0,
            .intermediates = 0,
    };
    for (const auto &layer : layers) {
        layer->createLayout(&counters);
    }

    uint32_t chainCount = inputImages.size();
    descriptorPool = createDescriptorPool(pLogicalDevice, std::vector<VkDescriptorPoolSize>(
            {
                    {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = counters.images},
                    {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = counters.uniforms},
                    {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = counters.intermediates},
            }
    ));
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
    std::vector<VkDescriptorImageInfo> imageInfos(2, {
            .sampler = VK_NULL_HANDLE,
            .imageView = VK_NULL_HANDLE,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    });
    std::vector<VkWriteDescriptorSet> writeDescriptorSets(imageInfos.size(), {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
    });
    for (uint32_t i = 0; i < imageInfos.size(); i++) {
        writeDescriptorSets[i].pImageInfo = &(imageInfos[i]);
    }

    Logger::debug("before writing descriptor Sets");
    VkDescriptorBufferInfo intermediateInfo{
            .buffer = VK_NULL_HANDLE,
            .offset = 0,
            .range = 0,
    };
    VkWriteDescriptorSet intermediateWrite = writeDescriptorSets[0];
    intermediateWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    intermediateWrite.dstBinding = 1;
    intermediateWrite.pImageInfo = nullptr;
    intermediateWrite.pBufferInfo = &intermediateInfo;
    VkDescriptorBufferInfo weightsInfo = intermediateInfo;
    weightsInfo.buffer = weights;
    VkWriteDescriptorSet weightsWrite = intermediateWrite;
    weightsWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    weightsWrite.dstBinding = 0;
    weightsWrite.pBufferInfo = &weightsInfo;
    auto holder = aist::DsWriterHolder{
            .inImage = &(writeDescriptorSets[0]),
            .outImage = &(writeDescriptorSets[1]),
            .intermediate = &intermediateWrite,
            .weights = &weightsWrite,
    };
    for (uint32_t chainIdx = 0; chainIdx < chainCount; chainIdx++) {
        imageInfos[0].imageView = inputImageViews[chainIdx];
        imageInfos[1].imageView = outputImageViews[chainIdx];
        intermediateInfo.buffer = intermediates[chainIdx];
        weightsInfo.offset = 0;
        weightsInfo.range = 0;
        for (const auto &layer : layers) {
            layer->writeSets(holder, chainIdx);
            weightsInfo.offset += weightsInfo.range;
            weightsInfo.range = 0;
            Logger::debug("Wrote DS in chain " + std::to_string(chainIdx));
        }
    }
    Logger::debug("after writing descriptor Sets");
}

void vkBasalt::AistEffect::applyEffect(uint32_t imageIndex, VkCommandBuffer commandBuffer) {
    Logger::debug("applying AistEffect to cb " + convertToString(commandBuffer));
    // After shader has run, modify layout of output image again to support present|transfer.
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

    // …and convert layout of output image because it was not modified after previous execution of this (same!) buffer.
    //TODO: Convert just before the last layer.
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

    VkBufferMemoryBarrier memoryBarrier{
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_MEMORY_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = intermediates[imageIndex],
            .offset = 0,
            .size = VK_WHOLE_SIZE,
    };
    bool addBufferBarrier = false;
    for (const auto &layer : layers) {
        if (addBufferBarrier) {
            pLogicalDevice->vkd.CmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0, nullptr,
                    1, &memoryBarrier,
                    0, nullptr
            );
        }
        layer->appendCommands(commandBuffer, imageIndex, &memoryBarrier);
        addBufferBarrier = true;
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
    pLogicalDevice->vkd.DestroyBuffer(pLogicalDevice->device, weights, nullptr);
    for (const auto &buffer : intermediates) {
        pLogicalDevice->vkd.DestroyBuffer(pLogicalDevice->device, buffer, nullptr);
    }
    pLogicalDevice->vkd.FreeMemory(pLogicalDevice->device, bufferMemory, nullptr);

    for (unsigned int i = 0; i < inputImageViews.size(); i++) {
        pLogicalDevice->vkd.DestroyImageView(pLogicalDevice->device, inputImageViews[i], nullptr);
        pLogicalDevice->vkd.DestroyImageView(pLogicalDevice->device, outputImageViews[i], nullptr);
    }
    Logger::debug("after DestroyImageView");
}

