#include <cmath>
#include <numeric>

#include "buffer.hpp"
#include "command_buffer.hpp"
#include "image_view.hpp"
#include "descriptor_set.hpp"
#include "util.hpp"
#include "memory.hpp"

#include "effect_aist.hpp"

namespace vkBasalt {
    const VkDeviceSize WEIGHT_ELEMENT_SIZE = 4;
    const VkDeviceSize W_STRIDED_LOW_SIZE = aist::LOW_STRIDED_CHANNELS * 1 * 10 * WEIGHT_ELEMENT_SIZE;
    const VkDeviceSize
        W_SHUFFLE_LOW_SIZE = (aist::LOW_SHUFFLE_CHANNELS * (aist::LOW_STRIDED_CHANNELS + 1)) * WEIGHT_ELEMENT_SIZE;
    const VkDeviceSize W_NORM_LOW_SIZE = aist::LOW_SHUFFLE_CHANNELS * 2 * WEIGHT_ELEMENT_SIZE;
    const VkDeviceSize W_CONV_HIGH_SIZE = aist::HIGH_CHANNELS * 1 * 10 * WEIGHT_ELEMENT_SIZE;
    const VkDeviceSize
        W_SHUFFLE_HIGH_SIZE = (aist::HIGH_CHANNELS * (aist::HIGH_CHANNELS + 1)) * WEIGHT_ELEMENT_SIZE;
    const VkDeviceSize W_NORM_HIGH_SIZE = aist::HIGH_CHANNELS * 2 * WEIGHT_ELEMENT_SIZE;
}

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
    lowExtent(
        {
            .width=(imageExtent.width + 1) / 2,
            .height=(imageExtent.height + 1) / 2,
        }
    ),
    highExtent(
        {
            .width=(lowExtent.width + 1) / 2,
            .height=(lowExtent.height + 1) / 2,
        }
    ),
    inputImages(inputImages),
    outputImages(outputImages),
    pConfig(pConfig),
    shaders(pLogicalDevice),
    // ⌈width / 2⌉ × ⌈height / 2⌉ × 16 × 2
    //   spatial|reduction   channels|   | 1) intermediate for split conv, 2) conv/norm target
    // or
    // ⌈width / 4⌉ × ⌈height / 4⌉ × 64 × 3
    //   spatial|reduction   channels|   | 1) identity passthrough, 2) intermediate for split conv, 3) conv/norm target.
    // w x h x 12 of 32-bit floats.
    //But we have to align offsets to 256 bytes boundary. Offsets will be at 4, 8.
    //x ≤ ⌈x / 2⌉ * 2 ≤ ⌈x / 4⌉ * 4
    //So, align ⌈width / 4⌉×⌈width / 4⌉×64 fp32 chunks.
    //(But something may spawn the area between them).
    //BTW, 64×4=256, so nothing to align actually.
    intermediateChunkAlignedSize(highExtent.width * highExtent.height * 64 * 4) {
    Logger::debug("in creating AistEffect");

    allocateBuffers();
    Logger::debug("AIST: allocated buffers");

    inputImageViews = createImageViews(pLogicalDevice, format, inputImages);
    Logger::debug("AIST: created input ImageViews");
    //Deal with crash during swapchain recreation. Likely, need to wait for active semaphores.
    //relayoutOutputImages();
    outputImageViews = createImageViews(pLogicalDevice, format, outputImages);
    Logger::debug("AIST: created output ImageViews");

    createDescriptorSets();
}

void vkBasalt::AistEffect::allocateBuffers() {
    auto weightsFileName = pConfig->getOption<std::string>("aistWeigthsFile");
    std::ifstream file(weightsFileName, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        Logger::err("Can't open aistWeigthsFile!");
    }
    auto fileSize = file.tellg();
    VkDeviceSize weightsSize = fileSize;
    char *weightsHostBuf = new char[fileSize];
    file.seekg(0, std::ios::beg);
    file.read(weightsHostBuf, fileSize);
    file.close();
    Logger::debug("Loaded aistWeigthsFile");
    VkBufferCreateInfo bufferInfo{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        //TODO: Remove once we start working with big enough weights.
        .size = std::max(weightsSize, W_SHUFFLE_HIGH_SIZE),
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkResult result = pLogicalDevice->vkd.CreateBuffer(pLogicalDevice->device, &bufferInfo, nullptr, &weights);
    ASSERT_VULKAN(result)
    //See constructor comment about this field.
    VkDeviceSize intermediateSize = intermediateChunkAlignedSize * 3;
    bufferInfo.size = intermediateSize;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    intermediates.resize(inputImages.size());
    for (auto &intermediate : intermediates) {
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

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingMemory;
    createBuffer(
        pLogicalDevice,
        weightsSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        stagingBuffer,
        stagingMemory
    );
    void *weightsCoherentBuf;
    result = pLogicalDevice->vkd
                           .MapMemory(pLogicalDevice->device, stagingMemory, 0, weightsSize, 0, &weightsCoherentBuf);
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
    pLogicalDevice->vkd.FreeCommandBuffers(
        pLogicalDevice->device, pLogicalDevice->commandPool, 1,
        &layoutCommandBuffer
    );
    Logger::trace("Relayout complete, extra freed");
}

void vkBasalt::AistEffect::createDescriptorSets() {
    VkResult result;
    uint32_t chainCount = inputImages.size();
    const uint32_t imagesDescriptorMultiplier = 2;
    const uint32_t buffersDescriptorMultiplier = 3;
    const uint32_t weightsDescriptorCount = 6;
    const auto &poolSizes = std::vector<VkDescriptorPoolSize>(
        {
            //in&out
            {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = chainCount * imagesDescriptorMultiplier},
            //for image, two halves, three thirds.
            {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = chainCount * buffersDescriptorMultiplier},
            //weights of six sizes
            {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = weightsDescriptorCount},
        }
    );
    descriptorPool = createDescriptorPool(pLogicalDevice, poolSizes);
    Logger::debug("AIST: created descriptorPool");

    //Can't join as sets have to be written to different holders.
    std::vector<VkDescriptorSetLayout> layouts;
    layouts.insert(layouts.cend(), chainCount * imagesDescriptorMultiplier, shaders.imageLayout);
    layouts.insert(layouts.cend(), chainCount * buffersDescriptorMultiplier, shaders.bufferLayout);
    layouts.insert(layouts.cend(), weightsDescriptorCount, shaders.weightsLayout);
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .pNext = nullptr,
        .descriptorPool = descriptorPool,
    };
    descriptorSetAllocateInfo.descriptorSetCount = layouts.size();
    descriptorSetAllocateInfo.pSetLayouts = layouts.data();
    std::vector<VkDescriptorSet> sets(layouts.size());
    result = pLogicalDevice->vkd.AllocateDescriptorSets(
        pLogicalDevice->device,
        &descriptorSetAllocateInfo,
        sets.data()
    );
    ASSERT_VULKAN(result)
    auto setsIt1 = sets.cbegin();
    auto setsIt2 = setsIt1 + chainCount;
    inputImageDescriptorSets.assign(setsIt1, setsIt2);
    setsIt1 = setsIt2 + chainCount;
    outputImageDescriptorSets.assign(setsIt2, setsIt1);
    setsIt2 = setsIt1 + chainCount;
    startThirdBufferDescriptorSets.assign(setsIt1, setsIt2);
    setsIt1 = setsIt2 + chainCount;
    midThirdBufferDescriptorSets.assign(setsIt2, setsIt1);
    setsIt2 = setsIt1 + chainCount;
    endThirdBufferDescriptorSets.assign(setsIt1, setsIt2);
    wStridedLowDescriptorSet = *(setsIt2++);
    wShuffleLowDescriptorSet = *(setsIt2++);
    wNormLowDescriptorSet = *(setsIt2++);
    wConvHighDescriptorSet = *(setsIt2++);
    wShuffleHighDescriptorSet = *(setsIt2++);
    wNormHighDescriptorSet = *(setsIt2++);
    Logger::debug("AIST: after allocating descriptor sets");

    Logger::debug("AIST: before writing descriptor sets");
    std::vector<VkDescriptorBufferInfo> bufferInfos(
        weightsDescriptorCount,
    {
            .buffer = weights,
            .offset = 0,
            .range = 0,
        }
    );
    std::vector<VkWriteDescriptorSet> writes(
        weightsDescriptorCount,
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .pImageInfo = nullptr,
            .pBufferInfo = nullptr,
            .pTexelBufferView = nullptr,
        }
    );
    for (size_t i = 0; i < weightsDescriptorCount; ++i) {
        writes[i].pBufferInfo = &bufferInfos[i];
    }
    bufferInfos[0].range = W_STRIDED_LOW_SIZE;
    writes[0].dstSet = wStridedLowDescriptorSet;
    bufferInfos[1].range = W_SHUFFLE_LOW_SIZE;
    writes[1].dstSet = wShuffleLowDescriptorSet;
    bufferInfos[2].range = W_CONV_HIGH_SIZE;
    writes[2].dstSet = wConvHighDescriptorSet;
    bufferInfos[3].range = W_SHUFFLE_HIGH_SIZE;
    writes[3].dstSet = wShuffleHighDescriptorSet;
    bufferInfos[4].range = W_NORM_LOW_SIZE;
    writes[4].dstSet = wNormLowDescriptorSet;
    bufferInfos[5].range = W_NORM_HIGH_SIZE;
    writes[5].dstSet = wNormHighDescriptorSet;
    pLogicalDevice->vkd.UpdateDescriptorSets(
        pLogicalDevice->device,
        writes.size(), writes.data(),
        0, nullptr
    );
    Logger::debug("AIST: weights DSes have been written to");

    std::vector<VkDescriptorImageInfo> imageInfos(
        imagesDescriptorMultiplier, {
            .sampler = VK_NULL_HANDLE,
            .imageView = VK_NULL_HANDLE,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        }
    );
    writes.resize(imagesDescriptorMultiplier + buffersDescriptorMultiplier);
    for (size_t i = 0; i < imagesDescriptorMultiplier; ++i) {
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pBufferInfo = nullptr;
        writes[i].pImageInfo = &imageInfos[i];
    }
    for (size_t i = imagesDescriptorMultiplier; i < imagesDescriptorMultiplier + buffersDescriptorMultiplier; ++i) {
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bufferInfos[i].range = intermediateChunkAlignedSize;
        bufferInfos[i].offset = (i - imagesDescriptorMultiplier) * intermediateChunkAlignedSize;
    }
    for (uint32_t chainIdx = 0; chainIdx < chainCount; chainIdx++) {
        imageInfos[0].imageView = inputImageViews[chainIdx];
        writes[0].dstSet = inputImageDescriptorSets[chainIdx];
        imageInfos[1].imageView = outputImageViews[chainIdx];
        writes[1].dstSet = outputImageDescriptorSets[chainIdx];
        for (size_t i = imagesDescriptorMultiplier; i < imagesDescriptorMultiplier + buffersDescriptorMultiplier; ++i) {
            bufferInfos[i].buffer = intermediates[chainIdx];
        }
        writes[2].dstSet = startThirdBufferDescriptorSets[chainIdx];
        writes[3].dstSet = midThirdBufferDescriptorSets[chainIdx];
        writes[4].dstSet = endThirdBufferDescriptorSets[chainIdx];
        pLogicalDevice->vkd.UpdateDescriptorSets(
            pLogicalDevice->device,
            writes.size(),
            writes.data(),
            0,
            nullptr
        );
        Logger::debug("AIST: Wrote DSes in chain " + std::to_string(chainIdx));
    }
    Logger::debug("AIST: after writing descriptor Sets");
}

void vkBasalt::AistEffect::applyEffect(uint32_t imageIndex, VkCommandBuffer commandBuffer) {
    Logger::debug("AIST: applying AistEffect to cb " + convertToString(commandBuffer));
    // After shader has run, modify layout of output image again to support present|transfer.
    VkImageMemoryBarrier outputAfterShaderBarrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = outputImages[imageIndex],
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        }
    };

    std::vector<VkImageMemoryBarrier> beforeShaderBarriers(2, outputAfterShaderBarrier);

    // Need to wait for previous effect and also convert layout…
    beforeShaderBarriers[0].srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    beforeShaderBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    beforeShaderBarriers[0].oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    beforeShaderBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    beforeShaderBarriers[1].image = inputImages[imageIndex];

    // …and convert layout of output image
    // because it was not modified after previous execution of this (same!) command buffer.
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
    Logger::debug("AIST: after the input pipeline barriers");

    VkBufferMemoryBarrier bufferBarrier{
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

    uint32_t weightsOffset = 0u;
    dispatchImageTouching(commandBuffer, imageIndex, false);

    appendBufferBarrier(commandBuffer, &bufferBarrier);
    dispatchStridedLow(commandBuffer, imageIndex, weightsOffset, false);
    weightsOffset += alignTo256Bytes(W_STRIDED_LOW_SIZE);
    appendBufferBarrier(commandBuffer, &bufferBarrier);
    dispatchShuffleLow(commandBuffer, imageIndex, weightsOffset, false);
    weightsOffset += alignTo256Bytes(W_SHUFFLE_LOW_SIZE);
    appendBufferBarrier(commandBuffer, &bufferBarrier);
    dispatchIn2D(commandBuffer, &bufferBarrier, imageIndex, weightsOffset);
    weightsOffset += alignTo256Bytes(W_NORM_LOW_SIZE);

    appendBufferBarrier(commandBuffer, &bufferBarrier);
    dispatchStridedHigh(commandBuffer, imageIndex, weightsOffset, false);
    weightsOffset += alignTo256Bytes(W_CONV_HIGH_SIZE);
    appendBufferBarrier(commandBuffer, &bufferBarrier);
    dispatchShuffleHigh(commandBuffer, imageIndex, weightsOffset, false);
    weightsOffset += alignTo256Bytes(W_SHUFFLE_HIGH_SIZE);

    appendBufferBarrier(commandBuffer, &bufferBarrier);
    dispatchShuffleHigh(commandBuffer, imageIndex, weightsOffset, true);
    weightsOffset += alignTo256Bytes(W_SHUFFLE_HIGH_SIZE);
    appendBufferBarrier(commandBuffer, &bufferBarrier);
    dispatchStridedHigh(commandBuffer, imageIndex, weightsOffset, true);
    weightsOffset += alignTo256Bytes(W_CONV_HIGH_SIZE);


    appendBufferBarrier(commandBuffer, &bufferBarrier);
    dispatchShuffleLow(commandBuffer, imageIndex, weightsOffset, true);
    weightsOffset += alignTo256Bytes(W_SHUFFLE_LOW_SIZE);
    appendBufferBarrier(commandBuffer, &bufferBarrier);
    dispatchStridedLow(commandBuffer, imageIndex, weightsOffset, true);
    weightsOffset += alignTo256Bytes(W_STRIDED_LOW_SIZE);//Not actually used, just let it be for code alignment.

    appendBufferBarrier(commandBuffer, &bufferBarrier);
    dispatchImageTouching(commandBuffer, imageIndex, true);

    pLogicalDevice->vkd.CmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &outputAfterShaderBarrier
    );

    Logger::debug("AIST: after the output pipeline barrier");
}

void vkBasalt::AistEffect::dispatchImageTouching(VkCommandBuffer commandBuffer, uint32_t chainIdx, bool output) {
    auto &shader = output ? shaders.toImage : shaders.fromImage;
    pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    VkDescriptorSet sets[]{
        output ? outputImageDescriptorSets[chainIdx] : inputImageDescriptorSets[chainIdx],
        startThirdBufferDescriptorSets[chainIdx],
    };
    pLogicalDevice->vkd.CmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        shader.layout, 0,
        std::size(sets), sets,
        0, nullptr
    );
    pLogicalDevice->vkd.CmdPushConstants(
        commandBuffer,
        shader.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(VkExtent2D),
        &imageExtent
    );
    pLogicalDevice->vkd.CmdDispatch(
        commandBuffer,
        (uint32_t) std::ceil((float) imageExtent.width / shader.scale),
        (uint32_t) std::ceil((float) imageExtent.height / shader.scale),
        shader.depthGlobals
    );
}

void vkBasalt::AistEffect::dispatchStridedLow(
    VkCommandBuffer commandBuffer,
    uint32_t chainIdx,
    uint32_t weightsOffset,
    bool up
) {
    auto &shader = up ? shaders.upConvLow : shaders.downConvLow;
    pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    VkDescriptorSet sets[]{
        startThirdBufferDescriptorSets[chainIdx],
        endThirdBufferDescriptorSets[chainIdx],
        wStridedLowDescriptorSet,
    };
    pLogicalDevice->vkd.CmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        shader.layout,
        0, std::size(sets), sets,
        1, &weightsOffset
    );
    pLogicalDevice->vkd.CmdPushConstants(
        commandBuffer, shader.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(VkExtent2D), &imageExtent
    );

    auto groupDrivingExtent = up ? &imageExtent : &lowExtent;
    pLogicalDevice->vkd.CmdDispatch(
        commandBuffer,
        (uint32_t) std::ceil((float) groupDrivingExtent->width / shader.scale),
        (uint32_t) std::ceil((float) groupDrivingExtent->height / shader.scale),
        shader.depthGlobals
    );
}

void vkBasalt::AistEffect::dispatchStridedHigh(
    VkCommandBuffer commandBuffer,
    uint32_t chainIdx,
    uint32_t weightsOffset,
    bool up
) {
    auto &shader = up ? shaders.upConvHigh : shaders.downConvHigh;
    pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    VkDescriptorSet sets[]{
        midThirdBufferDescriptorSets[chainIdx],
        endThirdBufferDescriptorSets[chainIdx],
        wConvHighDescriptorSet,
    };
    pLogicalDevice->vkd.CmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        shader.layout,
        0, std::size(sets), sets,
        1, &weightsOffset
    );
    pLogicalDevice->vkd.CmdPushConstants(
        commandBuffer, shader.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(VkExtent2D), &lowExtent
    );

    auto groupDrivingExtent = up ? &lowExtent : &highExtent;
    pLogicalDevice->vkd.CmdDispatch(
        commandBuffer,
        (uint32_t) std::ceil((float) groupDrivingExtent->width / shader.scale),
        (uint32_t) std::ceil((float) groupDrivingExtent->height / shader.scale),
        shader.depthGlobals
    );
}

void vkBasalt::AistEffect::dispatchShuffleLow(
    VkCommandBuffer commandBuffer,
    uint32_t chainIdx,
    uint32_t weightsOffset,
    bool up
) {
    auto &shader = up ? shaders.upShuffleLow : shaders.downShuffleLow;
    pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    VkDescriptorSet sets[]{
        up ? midThirdBufferDescriptorSets[chainIdx] : endThirdBufferDescriptorSets[chainIdx],
        up ? endThirdBufferDescriptorSets[chainIdx] : midThirdBufferDescriptorSets[chainIdx],
        wShuffleLowDescriptorSet,
    };
    pLogicalDevice->vkd.CmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        shader.layout,
        0, std::size(sets), sets,
        1, &weightsOffset
    );
    pLogicalDevice->vkd.CmdPushConstants(
        commandBuffer, shader.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(VkExtent2D), &lowExtent
    );

    pLogicalDevice->vkd.CmdDispatch(
        commandBuffer,
        (uint32_t) std::ceil((float) lowExtent.width * lowExtent.height / shader.scale),
        1u,
        shader.depthGlobals
    );
}

void vkBasalt::AistEffect::dispatchShuffleHigh(
    VkCommandBuffer commandBuffer,
    uint32_t chainIdx,
    uint32_t weightsOffset,
    bool inFromMid
) {
    auto &shader = shaders.shuffleHigh;
    pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    VkDescriptorSet sets[]{
        inFromMid ? midThirdBufferDescriptorSets[chainIdx] : endThirdBufferDescriptorSets[chainIdx],
        inFromMid ? endThirdBufferDescriptorSets[chainIdx] : midThirdBufferDescriptorSets[chainIdx],
        wShuffleHighDescriptorSet,
    };
    pLogicalDevice->vkd.CmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        shader.layout,
        0, std::size(sets), sets,
        1, &weightsOffset
    );
    pLogicalDevice->vkd.CmdPushConstants(
        commandBuffer, shader.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(VkExtent2D), &highExtent
    );

    pLogicalDevice->vkd.CmdDispatch(
        commandBuffer,
        (uint32_t) std::ceil((float) highExtent.width * highExtent.height / shader.scale),
        1u,
        shader.depthGlobals
    );
}

void vkBasalt::AistEffect::dispatchIn2D(
    VkCommandBuffer commandBuffer,
    VkBufferMemoryBarrier *bufferBarrierDto,
    uint32_t chainIdx,
    uint32_t weightsOffset
) {
    pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shaders.in2Dsum.pipeline);
    auto &characteristicShader = shaders.in2Dsum;
    VkDescriptorSet sets[]{
        midThirdBufferDescriptorSets[chainIdx],
        endThirdBufferDescriptorSets[chainIdx],
        wShuffleLowDescriptorSet,
    };
    pLogicalDevice->vkd.CmdBindDescriptorSets(
        commandBuffer,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        characteristicShader.layout,
        0, std::size(sets), sets,
        1, &weightsOffset
    );
    pLogicalDevice->vkd.CmdPushConstants(
        commandBuffer, characteristicShader.layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(VkExtent2D), &lowExtent
    );

    pLogicalDevice->vkd.CmdDispatch(commandBuffer, 1u, 1u, shaders.in2Dsum.depthGlobals);

    appendBufferBarrier(commandBuffer, bufferBarrierDto);
    pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shaders.in2Dcoeff.pipeline);
    pLogicalDevice->vkd.CmdDispatch(commandBuffer, 1u, 1u, shaders.in2Dcoeff.depthGlobals);

    appendBufferBarrier(commandBuffer, bufferBarrierDto);
    pLogicalDevice->vkd.CmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shaders.in2Dscale.pipeline);
    pLogicalDevice->vkd.CmdDispatch(
        commandBuffer,
        (uint32_t) std::ceil((float) lowExtent.width * lowExtent.height / shaders.in2Dscale.scale),
        1u,
        shaders.in2Dscale.depthGlobals
    );
}

void vkBasalt::AistEffect::appendBufferBarrier(VkCommandBuffer commandBuffer, VkBufferMemoryBarrier *bufferBarrierDto) {
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

VkDeviceSize vkBasalt::AistEffect::alignTo256Bytes(VkDeviceSize size) {
    VkDeviceSize overWholeWeight = size % 256;
    if (overWholeWeight > 0) {
        size += 256 - overWholeWeight;
    }
    return size;
}

vkBasalt::AistEffect::~AistEffect() {
    Logger::debug("destroying AistEffect " + convertToString(this));
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
    Logger::debug("AIST: after DestroyImageView");
}

