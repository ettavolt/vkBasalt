#include "nn_shaders.h"
#include "shader_source.h"

vkBasalt::aist::NnShaders::NnShaders(vkBasalt::LogicalDevice *pLogicalDevice) : pLogicalDevice(pLogicalDevice) {
    createSetLayouts();
    createShaders();
    createPipelineLayouts();
    createComputePipelines();
}

void vkBasalt::aist::NnShaders::createSetLayouts() {
    VkDescriptorSetLayoutBinding weightBinding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    };
    VkDescriptorSetLayoutCreateInfo descriptorSetCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = 1,
        .pBindings    = &weightBinding,
    };
    createSetLayout(&descriptorSetCreateInfo, &weightsLayout);

    VkDescriptorSetLayoutBinding storageBinding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    };
    descriptorSetCreateInfo.pBindings = &storageBinding;
    createSetLayout(&descriptorSetCreateInfo, &bufferLayout);

    storageBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    createSetLayout(&descriptorSetCreateInfo, &imageLayout);
}

void vkBasalt::aist::NnShaders::createSetLayout(
    VkDescriptorSetLayoutCreateInfo *createInfo,
    VkDescriptorSetLayout *handle
) {
    VkResult result = pLogicalDevice->vkd.CreateDescriptorSetLayout(
        pLogicalDevice->device,
        createInfo,
        nullptr,
        handle
    );
    ASSERT_VULKAN(result)
}

void vkBasalt::aist::NnShaders::createShaders() {
    VkShaderModuleCreateInfo shaderCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .pNext = nullptr,
        .flags = 0,
        .codeSize = sizeof(shaderSource::to_image), .pCode = shaderSource::to_image
    };
    createShader(&shaderCreateInfo, &toImage.module);

    shaderCreateInfo.codeSize = sizeof(shaderSource::from_image);
    shaderCreateInfo.pCode = shaderSource::from_image;
    createShader(&shaderCreateInfo, &fromImage.module);

    shaderCreateInfo.codeSize = sizeof(shaderSource::down_conv_low);
    shaderCreateInfo.pCode = shaderSource::down_conv_low;
    createShader(&shaderCreateInfo, &downConvLow.module);

    shaderCreateInfo.codeSize = sizeof(shaderSource::up_conv_low);
    shaderCreateInfo.pCode = shaderSource::up_conv_low;
    createShader(&shaderCreateInfo, &upConvLow.module);

    shaderCreateInfo.codeSize = sizeof(shaderSource::shuffle_low);
    shaderCreateInfo.pCode = shaderSource::shuffle_low;
    createShader(&shaderCreateInfo, &downShuffleLow.module);
    upShuffleLow.module = downShuffleLow.module;

    shaderCreateInfo.codeSize = sizeof(shaderSource::in_2d_sum_low);
    shaderCreateInfo.pCode = shaderSource::in_2d_sum_low;
    createShader(&shaderCreateInfo, &in2Dsum.module);

    shaderCreateInfo.codeSize = sizeof(shaderSource::in_2d_coeff_low);
    shaderCreateInfo.pCode = shaderSource::in_2d_coeff_low;
    createShader(&shaderCreateInfo, &in2Dcoeff.module);

    shaderCreateInfo.codeSize = sizeof(shaderSource::in_2d_scale_low);
    shaderCreateInfo.pCode = shaderSource::in_2d_scale_low;
    createShader(&shaderCreateInfo, &in2Dscale.module);
}

void vkBasalt::aist::NnShaders::createShader(VkShaderModuleCreateInfo *createInfo, VkShaderModule *handle) {
    VkResult result = pLogicalDevice->vkd.CreateShaderModule(
        pLogicalDevice->device,
        createInfo,
        nullptr,
        handle
    );
    ASSERT_VULKAN(result)
}

void vkBasalt::aist::NnShaders::createPipelineLayouts() {
    std::vector<VkDescriptorSetLayout> setLayouts(3, VK_NULL_HANDLE);
    setLayouts[0] = imageLayout;
    setLayouts[1] = bufferLayout;
    VkPushConstantRange pushConstantRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(VkExtent2D),
    };
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .setLayoutCount = 2,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pushConstantRange,
    };
    pipelineLayoutCreateInfo.pSetLayouts = setLayouts.data();
    //Maybe provide a getter on NnShader struct…
    createPipelineLayout(&pipelineLayoutCreateInfo, &imageBufferWidthHeightLayout);
    fromImage.layout = imageBufferWidthHeightLayout;
    toImage.layout = imageBufferWidthHeightLayout;

    setLayouts[0] = bufferLayout;
    setLayouts[2] = weightsLayout;
    pipelineLayoutCreateInfo.setLayoutCount = 3;
    createPipelineLayout(&pipelineLayoutCreateInfo, &twoBufferWeightsWidthHeightLayout);
    downConvLow.layout
        = upConvLow.layout
        = downShuffleLow.layout
        = upShuffleLow.layout
        = in2Dsum.layout
        = in2Dcoeff.layout
        = in2Dscale.layout
        = twoBufferWeightsWidthHeightLayout;
}

void vkBasalt::aist::NnShaders::createPipelineLayout(VkPipelineLayoutCreateInfo *createInfo, VkPipelineLayout *handle) {
    VkResult result = pLogicalDevice->vkd.CreatePipelineLayout(
        pLogicalDevice->device,
        createInfo,
        nullptr,
        handle
    );
    ASSERT_VULKAN(result)
}

void vkBasalt::aist::NnShaders::createComputePipelines() {
    uint32_t constIdx = 0;
    VkExtent3D specValues[]{
        {
            //Low-side channels
            .width = 0u,
            //High-side channels
            .height = 0u,
            //Other modifications
            .depth = 0u,
        },
        {
            .width = 0u,
            .height = 0u,
            .depth = 1u,
        },
    };
    VkSpecializationMapEntry specEntries[]{
        {.constantID = constIdx++, .offset = offsetof(VkExtent3D, width), .size = sizeof(VkExtent3D::width)},
        {.constantID = constIdx++, .offset = offsetof(VkExtent3D, height), .size = sizeof(VkExtent3D::height)},
        {.constantID = constIdx++, .offset = offsetof(VkExtent3D, depth), .size = sizeof(VkExtent3D::depth)},
        {
            .constantID = constIdx++,
            .offset = offsetof(VkExtent3D, width) + sizeof(VkExtent3D),
            .size = sizeof(VkExtent3D::width)
        },
        {
            .constantID = constIdx++,
            .offset = offsetof(VkExtent3D, height) + sizeof(VkExtent3D),
            .size = sizeof(VkExtent3D::height)
        },
        {
            .constantID = constIdx++,
            .offset = offsetof(VkExtent3D, depth) + sizeof(VkExtent3D),
            .size = sizeof(VkExtent3D::depth)
        },
    };
    VkSpecializationInfo specInfo{
        .mapEntryCount = constIdx, .pMapEntries = specEntries,
        .dataSize = sizeof(specValues), .pData = &specValues,
    };
    VkComputePipelineCreateInfo computePipelineCreateInfo{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = nullptr,
            .flags = 0,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = nullptr, .pName = "main",
            .pSpecializationInfo = &specInfo,
        },
        .layout = nullptr,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = -1,
    };
    specValues[1].width = specValues[1].height = fromImage.scale;
    createComputePipeline(&computePipelineCreateInfo, &fromImage);
    specValues[1].width = specValues[1].height = toImage.scale;
    createComputePipeline(&computePipelineCreateInfo, &toImage);

    specValues[0].width = IMAGE_CHANNELS;
    specValues[0].height = LOW_STRIDED_CHANNELS;
    specValues[1].width = specValues[1].height = upConvLow.scale;
    createComputePipeline(&computePipelineCreateInfo, &upConvLow);
    specValues[1].width = specValues[1].height = downConvLow.scale;
    specValues[1].depth = LOW_STRIDED_CHANNELS / downConvLow.depthGlobals;
    createComputePipeline(&computePipelineCreateInfo, &downConvLow);

    specValues[0].width = LOW_STRIDED_CHANNELS;
    specValues[0].height = LOW_SHUFFLE_CHANNELS;
    //Into low side
    specValues[0].depth = 0xFFFFFFFFu;
    specValues[1].height = 1u;
    specValues[1].width = upShuffleLow.scale;
    specValues[1].depth = LOW_STRIDED_CHANNELS;
    createComputePipeline(&computePipelineCreateInfo, &upShuffleLow);
    //Into high side
    specValues[0].depth = 0u;
    specValues[1].width = downShuffleLow.scale;
    specValues[1].depth = LOW_SHUFFLE_CHANNELS;
    createComputePipeline(&computePipelineCreateInfo, &downShuffleLow);

    specValues[0].width = specValues[0].height = LOW_SHUFFLE_CHANNELS;
    specValues[1].width = in2Dsum.scale;
    specValues[1].depth = 1u;
    createComputePipeline(&computePipelineCreateInfo, &in2Dsum);
    specValues[1].width = in2Dcoeff.scale;
    createComputePipeline(&computePipelineCreateInfo, &in2Dcoeff);
    specValues[1].width = in2Dscale.scale;
    createComputePipeline(&computePipelineCreateInfo, &in2Dscale);
}

void vkBasalt::aist::NnShaders::createComputePipeline(VkComputePipelineCreateInfo *createInfo, NnShader *shader) {
    createInfo->stage.module = shader->module;
    createInfo->layout = shader->layout;
    VkResult result = pLogicalDevice->vkd.CreateComputePipelines(
        pLogicalDevice->device,
        VK_NULL_HANDLE,
        1,
        createInfo,
        nullptr,
        &(shader->pipeline)
    );
    ASSERT_VULKAN(result)
}

vkBasalt::aist::NnShaders::~NnShaders() {
    destroyElements(&in2Dscale);
    destroyElements(&in2Dcoeff);
    destroyElements(&in2Dsum);
    //This is just a copy from downShuffleLow.
    upShuffleLow.module = VK_NULL_HANDLE;
    destroyElements(&upShuffleLow);
    destroyElements(&downShuffleLow);
    destroyElements(&upConvLow);
    destroyElements(&downConvLow);
    destroyElements(&toImage);
    destroyElements(&fromImage);
    if (twoBufferWeightsWidthHeightLayout != VK_NULL_HANDLE) {
        pLogicalDevice->vkd.DestroyPipelineLayout(pLogicalDevice->device, twoBufferWeightsWidthHeightLayout, nullptr);
    }
    if (imageBufferWidthHeightLayout != VK_NULL_HANDLE) {
        pLogicalDevice->vkd.DestroyPipelineLayout(pLogicalDevice->device, imageBufferWidthHeightLayout, nullptr);
    }
    destroySetLayout(bufferLayout);
    destroySetLayout(imageLayout);
    destroySetLayout(weightsLayout);
}

void vkBasalt::aist::NnShaders::destroyElements(NnShader *holder) {
    if (holder->pipeline != VK_NULL_HANDLE) {
        pLogicalDevice->vkd.DestroyPipeline(pLogicalDevice->device, holder->pipeline, nullptr);
    }
    if (holder->module != VK_NULL_HANDLE) {
        pLogicalDevice->vkd.DestroyShaderModule(pLogicalDevice->device, holder->module, nullptr);
    }
}

void vkBasalt::aist::NnShaders::destroySetLayout(VkDescriptorSetLayout setLayout) {
    if (setLayout != VK_NULL_HANDLE) {
        pLogicalDevice->vkd.DestroyDescriptorSetLayout(pLogicalDevice->device, setLayout, nullptr);
    }
}

