#include "application.hpp"
#include "matrices.hpp"

#include <imgui.h>
#include <lodepng.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <numbers>
#include <vector>

namespace application {
namespace {

struct Vertex {
    float position[3];
    float normal[3];
    float uv[2];
};

struct alignas(16) Uniforms {
    Mat4 mvp;
    Mat4 model;
    float camera_position[4];
    float light_direction[4];
    float light_color_intensity[4];
    float ambient_color_intensity[4];
    float albedo[4];
    float specular_shininess[4];
    float direction_to_camera[4];
};
static_assert(sizeof(Uniforms) == 240);
static_assert(offsetof(Uniforms, camera_position) == 128);
static_assert(offsetof(Uniforms, specular_shininess) == 208);
static_assert(offsetof(Uniforms, direction_to_camera) == 224);

struct Camera {
    float position[3] = { 0.0f, 0.0f, 5.0f };
    float rotation[3] = { 0.0f, 0.0f, 0.0f };
    float fov = 60.0f;
    float near_plane = 0.1f;
    float far_plane = 100.0f;
    float ortho_half_height = 2.5f;
};

struct Lighting {
    float azimuth = -35.0f;
    float elevation = -45.0f;
    float color[3] = { 1.0f, 1.0f, 1.0f };
    float intensity = 1.0f;
    float ambient_color[3] = { 1.0f, 1.0f, 1.0f };
    float ambient_intensity = 0.12f;
};

struct Material {
    float albedo[3] = { 1.0f, 1.0f, 1.0f };
    float specular[3] = { 0.7f, 0.7f, 0.7f };
    float shininess = 32.0f;
};

struct Model {
    float position[3] = { 0.0f, 0.0f, 0.0f };
    float rotation[3] = { 0.35f, 0.0f, 0.0f };
    float scale = 1.0f;
    Material material;
};

const std::array<Model, 2> initial_models = {
    Model{},
    Model{ .position = { 2.4f, 0.0f, -0.8f }, .scale = 0.8f },
};
std::array<Model, 2> models = initial_models;
int selected_model = 0;
Camera camera;
Lighting lighting;

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    void* mapped = nullptr;
};

struct Texture {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
};

constexpr uint32_t segments = 32;
constexpr float pi = std::numbers::pi_v<float>;
Buffer vertex_buffer, index_buffer, uniform_buffer;
Texture texture;
VkDeviceSize uniform_stride = 0;
VkCommandPool upload_pool = VK_NULL_HANDLE;
uint32_t index_count = 0;
VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

bool use_perspective = true;

bool check(VkResult result, const char* operation) {
    if (result == VK_SUCCESS) return true;
    std::cerr << operation << " failed: " << result << '\n';
    return false;
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer,
                  bool device_local = false) {
    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation_info = {
        .flags = device_local ? 0u : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                    VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = device_local ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE : VMA_MEMORY_USAGE_AUTO,
        .requiredFlags = device_local ? VkMemoryPropertyFlags(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) : 0u,
    };
    VmaAllocationInfo allocation{};
    if (!check(vmaCreateBuffer(graphics::internal::context.allocator, &info,
            &allocation_info, &buffer.handle, &buffer.allocation, &allocation),
            "vmaCreateBuffer")) return false;
    buffer.mapped = allocation.pMappedData;
    return true;
}

bool upload(const Buffer& buffer, const void* data, size_t size, VkDeviceSize offset = 0) {
    std::memcpy(static_cast<std::byte*>(buffer.mapped) + offset, data, size);
    return check(vmaFlushAllocation(graphics::internal::context.allocator,
        buffer.allocation, offset, size), "vmaFlushAllocation");
}

bool createUploadPool() {
    const auto& context = graphics::internal::context;
    const VkCommandPoolCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = context.graphics_queue_index,
    };
    return check(vkCreateCommandPool(context.device, &info, nullptr, &upload_pool),
        "vkCreateCommandPool");
}

bool beginUpload(VkCommandBuffer& command_buffer) {
    const auto device = graphics::internal::context.device;
    const VkCommandBufferAllocateInfo allocation = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = upload_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (!check(vkAllocateCommandBuffers(device, &allocation, &command_buffer),
            "vkAllocateCommandBuffers")) return false;
    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    if (check(vkBeginCommandBuffer(command_buffer, &begin), "vkBeginCommandBuffer")) return true;
    vkFreeCommandBuffers(device, upload_pool, 1, &command_buffer);
    return false;
}

bool finishUpload(VkCommandBuffer command_buffer) {
    const auto& context = graphics::internal::context;
    const VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };
    const bool result = check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer") &&
        check(vkQueueSubmit(context.graphics_queue, 1, &submit, VK_NULL_HANDLE), "vkQueueSubmit") &&
        check(vkQueueWaitIdle(context.graphics_queue), "vkQueueWaitIdle");
    vkFreeCommandBuffers(context.device, upload_pool, 1, &command_buffer);
    return result;
}

bool createTexture(const std::filesystem::path& path) {
    const auto& context = graphics::internal::context;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "Cannot open texture: " << path << '\n';
        return false;
    }
    const std::vector<unsigned char> encoded((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    std::vector<unsigned char> pixels;
    unsigned width = 0, height = 0;
    const unsigned error = lodepng::decode(pixels, width, height, encoded);
    if (error) {
        std::cerr << "Cannot decode texture: " << lodepng_error_text(error) << '\n';
        return false;
    }
    const VkImageCreateInfo image_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_SRGB,
        .extent = { width, height, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VmaAllocationCreateInfo allocation = {
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    };
    if (!check(vmaCreateImage(context.allocator, &image_info, &allocation,
            &texture.image, &texture.allocation, nullptr), "vmaCreateImage")) return false;
    const VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = texture.image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = image_info.format,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 },
    };
    if (!check(vkCreateImageView(context.device, &view_info, nullptr, &texture.view),
            "vkCreateImageView")) return false;
    const VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
    };
    if (!check(vkCreateSampler(context.device, &sampler_info, nullptr, &texture.sampler),
            "vkCreateSampler")) return false;

    Buffer staging;
    if (!createBuffer(pixels.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) return false;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    bool result = upload(staging, pixels.data(), pixels.size()) && beginUpload(command_buffer);
    if (result) {
        VkImageMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = texture.image,
            .subresourceRange = view_info.subresourceRange,
        };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        const VkBufferImageCopy region = {
            .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
            .imageExtent = { width, height, 1 },
        };
        vkCmdCopyBufferToImage(command_buffer, staging.handle, texture.image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        result = finishUpload(command_buffer);
    }
    vmaDestroyBuffer(context.allocator, staging.handle, staging.allocation);
    return result;
}

bool createCone() {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    const auto addTriangle = [&](const std::array<float, 3>& a,
                                 const std::array<float, 3>& b,
                                 const std::array<float, 3>& c,
                                 const std::array<std::array<float, 2>, 3>& uv) {
        const float ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
        const float vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
        float nx = uy * vz - uz * vy;
        float ny = uz * vx - ux * vz;
        float nz = ux * vy - uy * vx;
        const float length = std::sqrt(nx * nx + ny * ny + nz * nz);
        nx /= length;
        ny /= length;
        nz /= length;
        const std::array points = { a, b, c };
        for (size_t j = 0; j < points.size(); ++j) {
            const auto& p = points[j];
            indices.push_back(uint32_t(vertices.size()));
            vertices.push_back({ { p[0], p[1], p[2] }, { nx, ny, nz }, { uv[j][0], uv[j][1] } });
        }
    };
    for (uint32_t i = 0; i < segments; ++i) {
        const float angle = 2.0f * pi * float(i) / float(segments);
        const float next_angle = 2.0f * pi * float((i + 1) % segments) / float(segments);
        const std::array<float, 3> current = { std::cos(angle), -1.0f, std::sin(angle) };
        const std::array<float, 3> next = { std::cos(next_angle), -1.0f, std::sin(next_angle) };
        const float u = float(i) / float(segments);
        const float next_u = float(i + 1) / float(segments);
        addTriangle({ 0.0f, 1.0f, 0.0f }, next, current,
            {{ { (u + next_u) * 0.5f, 0.0f }, { next_u, 1.0f }, { u, 1.0f } }});
        addTriangle({ 0.0f, -1.0f, 0.0f }, current, next,
            {{ { 0.5f, 0.5f }, { 0.5f + current[0] * 0.5f, 0.5f + current[2] * 0.5f },
               { 0.5f + next[0] * 0.5f, 0.5f + next[2] * 0.5f } }});
    }
    index_count = uint32_t(indices.size());
    const VkDeviceSize vertex_size = vertices.size() * sizeof(Vertex);
    const VkDeviceSize index_size = indices.size() * sizeof(uint32_t);
    if (!createBuffer(vertex_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vertex_buffer, true) ||
        !createBuffer(index_size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            index_buffer, true)) return false;
    Buffer staging;
    if (!createBuffer(vertex_size + index_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging)) return false;
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    bool result = upload(staging, vertices.data(), size_t(vertex_size)) &&
        upload(staging, indices.data(), size_t(index_size), vertex_size) && beginUpload(command_buffer);
    if (result) {
        const VkBufferCopy vertex_copy = { 0, 0, vertex_size };
        const VkBufferCopy index_copy = { vertex_size, 0, index_size };
        vkCmdCopyBuffer(command_buffer, staging.handle, vertex_buffer.handle, 1, &vertex_copy);
        vkCmdCopyBuffer(command_buffer, staging.handle, index_buffer.handle, 1, &index_copy);
        const VkBufferMemoryBarrier barriers[] = {
            { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
              .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
              .dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .buffer = vertex_buffer.handle, .size = VK_WHOLE_SIZE },
            { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
              .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
              .dstAccessMask = VK_ACCESS_INDEX_READ_BIT,
              .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
              .buffer = index_buffer.handle, .size = VK_WHOLE_SIZE },
        };
        vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr, 2, barriers, 0, nullptr);
        result = finishUpload(command_buffer);
    }
    vmaDestroyBuffer(graphics::internal::context.allocator, staging.handle, staging.allocation);
    return result;
}

bool createUniforms() {
    const auto device = graphics::internal::context.device;
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &properties);
    const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
    uniform_stride = (sizeof(Uniforms) + alignment - 1) & ~(alignment - 1);
    if (!createBuffer(uniform_stride * models.size(), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            uniform_buffer)) return false;
    const VkDescriptorSetLayoutBinding bindings[] = {
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT },
    };
    const VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = bindings,
    };
    if (!check(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_layout),
            "vkCreateDescriptorSetLayout")) return false;
    const VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
    };
    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
    };
    if (!check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool),
            "vkCreateDescriptorPool")) return false;
    const VkDescriptorSetAllocateInfo allocate_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_layout,
    };
    if (!check(vkAllocateDescriptorSets(device, &allocate_info, &descriptor_set),
            "vkAllocateDescriptorSets")) return false;
    const VkDescriptorBufferInfo buffer_info = { uniform_buffer.handle, 0, sizeof(Uniforms) };
    const VkDescriptorImageInfo image_info = {
        texture.sampler, texture.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkWriteDescriptorSet writes[] = {
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set,
          .dstBinding = 0, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &buffer_info },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = descriptor_set,
          .dstBinding = 1, .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &image_info },
    };
    vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
    return true;
}

VkShaderModule loadShader(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Cannot open shader: " << path << '\n';
        return VK_NULL_HANDLE;
    }
    const auto size = file.tellg();
    if (size <= 0 || size % 4 != 0) {
        std::cerr << "Invalid SPIR-V file: " << path << '\n';
        return VK_NULL_HANDLE;
    }
    std::vector<uint32_t> code(size_t(size) / sizeof(uint32_t));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(code.data()), size)) return VK_NULL_HANDLE;
    const VkShaderModuleCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size_t(size),
        .pCode = code.data(),
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    check(vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &shader),
        "vkCreateShaderModule");
    return shader;
}

bool createPipeline(const std::filesystem::path& shader_directory) {
    const auto& context = graphics::internal::context;
    const VkPipelineLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_layout,
    };
    if (!check(vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout),
            "vkCreatePipelineLayout")) return false;
    const VkShaderModule vertex = loadShader(shader_directory / "cone.vert.spv");
    const VkShaderModule fragment = loadShader(shader_directory / "cone.frag.spv");
    if (!vertex || !fragment) {
        vkDestroyShaderModule(context.device, vertex, nullptr);
        vkDestroyShaderModule(context.device, fragment, nullptr);
        return false;
    }
    const VkPipelineShaderStageCreateInfo stages[] = {
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertex, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragment, .pName = "main" },
    };
    const VkVertexInputBindingDescription binding = { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
    const VkVertexInputAttributeDescription attributes[] = {
        { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, position) },
        { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
        { 2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv) },
    };
    const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 3, .pVertexAttributeDescriptions = attributes,
    };
    const VkPipelineInputAssemblyStateCreateInfo assembly = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const VkPipelineViewportStateCreateInfo viewport = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1, .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo rasterization = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisampling = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const VkPipelineDepthStencilStateCreateInfo depth = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS,
    };
    const VkPipelineColorBlendAttachmentState color_attachment = {
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo color_blend = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1, .pAttachments = &color_attachment,
    };
    const VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    const VkPipelineDynamicStateCreateInfo dynamic = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = 2, .pDynamicStates = dynamic_states,
    };
    const VkGraphicsPipelineCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = 2, .pStages = stages,
        .pVertexInputState = &vertex_input,
        .pInputAssemblyState = &assembly,
        .pViewportState = &viewport,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisampling,
        .pDepthStencilState = &depth,
        .pColorBlendState = &color_blend,
        .pDynamicState = &dynamic,
        .layout = pipeline_layout,
        .renderPass = context.render_pass,
        .subpass = 0,
    };
    const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE,
        1, &info, nullptr, &pipeline);
    vkDestroyShaderModule(context.device, fragment, nullptr);
    vkDestroyShaderModule(context.device, vertex, nullptr);
    return check(result, "vkCreateGraphicsPipelines");
}

void resetScene() {
    models = initial_models;
    use_perspective = true;
    camera = Camera{};
    lighting = Lighting{};
}

} // namespace

bool initialize(const std::filesystem::path& shader_directory) {
    if (createUploadPool() && createCone() &&
        createTexture(shader_directory.parent_path() / "textures" / "checker.png") &&
        createUniforms() && createPipeline(shader_directory)) return true;
    shutdown();
    return false;
}

void shutdown() {
    const auto& context = graphics::internal::context;
    vkDeviceWaitIdle(context.device);
    vkDestroyPipeline(context.device, pipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
    vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, descriptor_layout, nullptr);
    vkDestroySampler(context.device, texture.sampler, nullptr);
    vkDestroyImageView(context.device, texture.view, nullptr);
    if (texture.image) vmaDestroyImage(context.allocator, texture.image, texture.allocation);
    vkDestroyCommandPool(context.device, upload_pool, nullptr);
    for (const Buffer* buffer : { &uniform_buffer, &index_buffer, &vertex_buffer })
        if (buffer->handle) vmaDestroyBuffer(context.allocator, buffer->handle, buffer->allocation);
}

void update([[maybe_unused]] double time) {
    const auto& io = ImGui::GetIO();
    const float step = std::min(io.DeltaTime, 0.1f);
    ImGui::SetNextWindowPos(ImVec2(15, 15), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420, std::min(665.0f, io.DisplaySize.y - 30.0f)), ImGuiCond_Always);
    ImGui::Begin("Lab 3 - Texture mapping", nullptr, ImGuiWindowFlags_NoResize);
    ImGui::TextUnformatted("Variant 9: cone");
    ImGui::Combo("Object", &selected_model, "Cone 1\0Cone 2\0");
    auto& position = models[selected_model].position;
    auto& rotation = models[selected_model].rotation;
    float& scale = models[selected_model].scale;
    auto& material = models[selected_model].material;
    if (!io.WantCaptureKeyboard) {
        if (ImGui::IsKeyPressed(ImGuiKey_P, false)) use_perspective = !use_perspective;
        if (ImGui::IsKeyPressed(ImGuiKey_R, false)) resetScene();
        if (ImGui::IsKeyDown(ImGuiKey_LeftArrow)) rotation[1] -= step;
        if (ImGui::IsKeyDown(ImGuiKey_RightArrow)) rotation[1] += step;
        if (ImGui::IsKeyDown(ImGuiKey_UpArrow)) rotation[0] -= step;
        if (ImGui::IsKeyDown(ImGuiKey_DownArrow)) rotation[0] += step;
        if (ImGui::IsKeyDown(ImGuiKey_Z)) rotation[2] -= step;
        if (ImGui::IsKeyDown(ImGuiKey_C)) rotation[2] += step;
        if (ImGui::IsKeyDown(ImGuiKey_A)) position[0] -= step;
        if (ImGui::IsKeyDown(ImGuiKey_D)) position[0] += step;
        if (ImGui::IsKeyDown(ImGuiKey_W)) position[1] += step;
        if (ImGui::IsKeyDown(ImGuiKey_S)) position[1] -= step;
        if (ImGui::IsKeyDown(ImGuiKey_Q)) position[2] -= step;
        if (ImGui::IsKeyDown(ImGuiKey_E)) position[2] += step;
        if (ImGui::IsKeyDown(ImGuiKey_Equal) || ImGui::IsKeyDown(ImGuiKey_KeypadAdd)) scale += step;
        if (ImGui::IsKeyDown(ImGuiKey_Minus) || ImGui::IsKeyDown(ImGuiKey_KeypadSubtract)) scale -= step;
        scale = std::clamp(scale, 0.1f, 3.0f);
        for (float& value : position) value = std::clamp(value, -3.0f, 3.0f);
    }
    if (ImGui::Button("Reset scene (R)")) resetScene();
    ImGui::TextUnformatted("Drag values or Ctrl+click to enter a number.");
    ImGui::PushItemWidth(245.0f);
    constexpr auto clamp = ImGuiSliderFlags_AlwaysClamp;
    if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("Position", camera.position, 0.02f, -20.0f, 20.0f, "%.2f", clamp);
        ImGui::DragFloat3("Pitch/Yaw/Roll", camera.rotation, 0.5f, -180.0f, 180.0f, "%.1f", clamp);
        ImGui::Checkbox("Perspective (P)", &use_perspective);
        ImGui::SliderFloat("FOV (degrees)", &camera.fov, 20.0f, 120.0f, "%.1f", clamp);
        ImGui::DragFloat("Near plane", &camera.near_plane, 0.01f, 0.01f, camera.far_plane - 0.01f, "%.2f", clamp);
        ImGui::DragFloat("Far plane", &camera.far_plane, 0.1f, camera.near_plane + 0.01f, 500.0f, "%.2f", clamp);
        if (!use_perspective)
            ImGui::SliderFloat("Ortho size", &camera.ortho_half_height, 0.2f, 10.0f, "%.2f", clamp);
    }
    if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::SliderFloat("Azimuth", &lighting.azimuth, -180.0f, 180.0f, "%.1f deg", clamp);
        ImGui::SliderFloat("Elevation", &lighting.elevation, -90.0f, 90.0f, "%.1f deg", clamp);
        ImGui::ColorEdit3("Light color", lighting.color);
        ImGui::SliderFloat("Intensity", &lighting.intensity, 0.0f, 3.0f, "%.2f", clamp);
        ImGui::TextUnformatted("Angles set the direction in which rays travel.");
    }
    if (ImGui::CollapsingHeader("Ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::ColorEdit3("Ambient color", lighting.ambient_color);
        ImGui::SliderFloat("Ambient intensity", &lighting.ambient_intensity, 0.0f, 1.0f, "%.2f", clamp);
    }
    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::TextUnformatted("Texture: checker.png");
        ImGui::ColorEdit3("Albedo", material.albedo);
        ImGui::ColorEdit3("Specular color", material.specular);
        ImGui::SliderFloat("Shininess", &material.shininess, 1.0f, 256.0f, "%.1f", clamp);
    }
    if (ImGui::CollapsingHeader("Model")) {
        ImGui::DragFloat3("Translation", position, 0.02f, -3.0f, 3.0f, "%.2f", clamp);
        ImGui::SliderAngle("Rotation X", &rotation[0], -180.0f, 180.0f, "%.1f deg", clamp);
        ImGui::SliderAngle("Rotation Y", &rotation[1], -180.0f, 180.0f, "%.1f deg", clamp);
        ImGui::SliderAngle("Rotation Z", &rotation[2], -180.0f, 180.0f, "%.1f deg", clamp);
        ImGui::SliderFloat("Scale", &scale, 0.1f, 3.0f, "%.2f", clamp);
        ImGui::TextUnformatted("Arrows: rotate X/Y | Z/C: rotate Z");
        ImGui::TextUnformatted("W/A/S/D: move X/Y | Q/E: move Z | +/-: scale");
    }
    ImGui::PopItemWidth();
    ImGui::Text("%u objects | %u triangles | %.0f FPS", unsigned(models.size()),
        unsigned(models.size()) * index_count / 3, io.Framerate);
    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    const auto& context = graphics::internal::context;
    const float aspect = float(context.swapchain_extent.width) / float(context.swapchain_extent.height);
    const float radians = pi / 180.0f;
    const Mat4 view = viewMatrix(camera.position[0], camera.position[1], camera.position[2],
        camera.rotation[0] * radians, camera.rotation[1] * radians, camera.rotation[2] * radians);
    const Mat4 projection = use_perspective ?
        perspective(camera.fov * radians, aspect, camera.near_plane, camera.far_plane) :
        orthographic(camera.ortho_half_height, aspect, camera.near_plane, camera.far_plane);
    const float azimuth = lighting.azimuth * radians;
    const float elevation = lighting.elevation * radians;
    for (size_t i = 0; i < models.size(); ++i) {
        const auto& object = models[i];
        const auto& material = object.material;
        const Mat4 model = translation(object.position[0], object.position[1], object.position[2]) *
            rotationZ(object.rotation[2]) * rotationY(object.rotation[1]) *
            rotationX(object.rotation[0]) * scaling(object.scale);
        const Uniforms uniforms = {
            projection * view * model,
            model,
            { camera.position[0], camera.position[1], camera.position[2], use_perspective ? 1.0f : 0.0f },
            { std::cos(elevation) * std::sin(azimuth), std::sin(elevation),
              -std::cos(elevation) * std::cos(azimuth), 0.0f },
            { lighting.color[0], lighting.color[1], lighting.color[2], lighting.intensity },
            { lighting.ambient_color[0], lighting.ambient_color[1], lighting.ambient_color[2], lighting.ambient_intensity },
            { material.albedo[0], material.albedo[1], material.albedo[2], 0.0f },
            { material.specular[0], material.specular[1], material.specular[2], material.shininess },
            { view.at(2, 0), view.at(2, 1), view.at(2, 2), 0.0f },
        };
        upload(uniform_buffer, &uniforms, sizeof(uniforms), i * uniform_stride);
    }

    vkResetCommandBuffer(fd.command_buffer, 0);
    const VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(fd.command_buffer, &begin);
    const VkClearValue clear_values[] = {
        { .color = { .float32 = { 0.1f, 0.1f, 0.15f, 1.0f } } },
        { .depthStencil = { 1.0f, 0 } },
    };
    const VkRenderPassBeginInfo render_pass = {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = context.render_pass,
        .framebuffer = fd.framebuffer,
        .renderArea = { .extent = context.swapchain_extent },
        .clearValueCount = 2,
        .pClearValues = clear_values,
    };
    vkCmdBeginRenderPass(fd.command_buffer, &render_pass, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport = { 0.0f, 0.0f, float(context.swapchain_extent.width),
        float(context.swapchain_extent.height), 0.0f, 1.0f };
    const VkRect2D scissor = { { 0, 0 }, context.swapchain_extent };
    vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
    vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
    vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.handle, &offset);
    vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);
    for (size_t i = 0; i < models.size(); ++i) {
        const uint32_t dynamic_offset = uint32_t(i * uniform_stride);
        vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout, 0, 1, &descriptor_set, 1, &dynamic_offset);
        vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
    }
    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
