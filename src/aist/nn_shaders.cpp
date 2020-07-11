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

void vkBasalt::aist::NnShaders::createSetLayout(VkDescriptorSetLayoutCreateInfo *createInfo, VkDescriptorSetLayout *handle) {
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
    std::vector<VkDescriptorSetLayout> setLayouts(2, VK_NULL_HANDLE);
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
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &pushConstantRange,
    };
    pipelineLayoutCreateInfo.setLayoutCount = setLayouts.size();
    pipelineLayoutCreateInfo.pSetLayouts = setLayouts.data();
    createPipelineLayout(&pipelineLayoutCreateInfo, &fromImage.layout);
    createPipelineLayout(&pipelineLayoutCreateInfo, &toImage.layout);
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
        .width = imageAccessColumnGroup,
        .height = imageAccessRowGroup,
        .depth = 1,
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
                    .module = fromImage.module, .pName = "main",
                    .pSpecializationInfo = &specInfo,
            },
            .layout = fromImage.layout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
    };
    createComputePipeline(&computePipelineCreateInfo, &fromImage.pipeline);

    computePipelineCreateInfo.stage.module = toImage.module;
    computePipelineCreateInfo.layout = toImage.layout;
    createComputePipeline(&computePipelineCreateInfo, &toImage.pipeline);
}

void vkBasalt::aist::NnShaders::createComputePipeline(VkComputePipelineCreateInfo *createInfo, VkPipeline *handle) {
    VkResult result = pLogicalDevice->vkd.CreateComputePipelines(
            pLogicalDevice->device,
            VK_NULL_HANDLE,
            1,
            createInfo,
            nullptr,
            handle
    );
    ASSERT_VULKAN(result)
}

vkBasalt::aist::NnShaders::~NnShaders() {
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

uint32_t vkBasalt::aist::NnShaders::getImageAccessColumnGroup() const {
    return imageAccessColumnGroup;
}

uint32_t vkBasalt::aist::NnShaders::getImageAccessRowGroup() const {
    return imageAccessRowGroup;
}
