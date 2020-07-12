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
    //TODO: consider deduplication.
    //Store same on this, copy pointer to shaders.
    //Destruct on this, respectively.
    //Maybe provide a getter…
    createPipelineLayout(&pipelineLayoutCreateInfo, &fromImage.layout);
    createPipelineLayout(&pipelineLayoutCreateInfo, &toImage.layout);

    setLayouts[0] = bufferLayout;
    setLayouts[2] = weightsLayout;
    pipelineLayoutCreateInfo.setLayoutCount = 3;
    createPipelineLayout(&pipelineLayoutCreateInfo, &downConvLow.layout);
    createPipelineLayout(&pipelineLayoutCreateInfo, &upConvLow.layout);
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
    VkSpecializationMapEntry specEntries[]{
        //NVidia can't bind compute group size to zeroth contstant
        {.constantID = constIdx++, .offset = offsetof(VkExtent3D, width), .size = sizeof(VkExtent3D::width)},
        {.constantID = constIdx++, .offset = offsetof(VkExtent3D, width), .size = sizeof(VkExtent3D::width)},
        {.constantID = constIdx++, .offset = offsetof(VkExtent3D, height), .size = sizeof(VkExtent3D::height)},
        {.constantID = constIdx++, .offset = offsetof(VkExtent3D, depth), .size = sizeof(VkExtent3D::depth)},
    };
    VkExtent3D groupSizes{
        .width = 0u,
        .height = 0u,
        .depth = 1u,
    };
    VkSpecializationInfo specInfo{
        .mapEntryCount = constIdx, .pMapEntries = specEntries,
        .dataSize = sizeof(groupSizes), .pData = &groupSizes,
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
    groupSizes.width = groupSizes.height = fromImage.scale;
    createComputePipeline(&computePipelineCreateInfo, &fromImage);

    groupSizes.width = groupSizes.height = toImage.scale;
    createComputePipeline(&computePipelineCreateInfo, &toImage);

    groupSizes.width = groupSizes.height = upConvLow.scale;
    createComputePipeline(&computePipelineCreateInfo, &upConvLow);

    groupSizes.width = groupSizes.height = downConvLow.scale;
    groupSizes.depth = LOW_STRIDED_CHANNELS / downConvLow.depthGlobals;
    createComputePipeline(&computePipelineCreateInfo, &downConvLow);
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
    destroyElements(&upConvLow);
    destroyElements(&downConvLow);
    destroyElements(&toImage);
    destroyElements(&fromImage);
    destroySetLayout(bufferLayout);
    destroySetLayout(imageLayout);
    destroySetLayout(weightsLayout);
}

void vkBasalt::aist::NnShaders::destroyElements(NnShader *holder) {
    if (holder->pipeline != VK_NULL_HANDLE) {
        pLogicalDevice->vkd.DestroyPipeline(pLogicalDevice->device, holder->pipeline, nullptr);
    }
    if (holder->layout != VK_NULL_HANDLE) {
        pLogicalDevice->vkd.DestroyPipelineLayout(pLogicalDevice->device, holder->layout, nullptr);
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

