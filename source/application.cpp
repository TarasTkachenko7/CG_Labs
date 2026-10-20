#include "application.hpp"
#include "matrices.hpp"

#include <imgui.h>
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numbers>
#include <vector>

namespace application {
namespace {

struct Vertex {
    float position[3];
    float normal[3];
};

// vec4 fields keep the CPU layout identical to std140 in both shaders.
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
    float rotation[3] = { 0.0f, 0.0f, 0.0f }; // pitch, yaw, roll in degrees
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
    float albedo[3] = { 0.95f, 0.38f, 0.08f };
    float specular[3] = { 0.7f, 0.7f, 0.7f };
    float shininess = 32.0f;
};

Camera camera;
Lighting lighting;
Material material;

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    void* mapped = nullptr;
};

constexpr uint32_t segments = 32;
constexpr float pi = std::numbers::pi_v<float>;
Buffer vertex_buffer, index_buffer, uniform_buffer;
uint32_t index_count = 0;
VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

float position[3] = { 0.0f, 0.0f, 0.0f };
float rotation[3] = { 0.35f, 0.0f, 0.0f };
float scale = 1.0f;
bool use_perspective = true;

bool check(VkResult result, const char* operation) {
    if (result == VK_SUCCESS) return true;
    std::cerr << operation << " failed: " << result << '\n';
    return false;
}

bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer) {
    const VkBufferCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo allocation_info = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VmaAllocationInfo allocation{};
    if (!check(vmaCreateBuffer(graphics::internal::context.allocator, &info,
            &allocation_info, &buffer.handle, &buffer.allocation, &allocation),
            "vmaCreateBuffer")) return false;
    buffer.mapped = allocation.pMappedData;
    return true;
}

bool upload(const Buffer& buffer, const void* data, size_t size) {
    std::memcpy(buffer.mapped, data, size);
    return check(vmaFlushAllocation(graphics::internal::context.allocator,
        buffer.allocation, 0, size), "vmaFlushAllocation");
}

bool createCone() {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    // Each triangle has its own vertices: the side and base need different normals.
    const auto addTriangle = [&](const std::array<float, 3>& a,
                                 const std::array<float, 3>& b,
                                 const std::array<float, 3>& c) {
        const float ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
        const float vx = c[0] - a[0], vy = c[1] - a[1], vz = c[2] - a[2];
        float nx = uy * vz - uz * vy;
        float ny = uz * vx - ux * vz;
        float nz = ux * vy - uy * vx;
        const float length = std::sqrt(nx * nx + ny * ny + nz * nz);
        nx /= length;
        ny /= length;
        nz /= length;
        for (const auto& p : { a, b, c }) {
            indices.push_back(uint32_t(vertices.size()));
            vertices.push_back({ { p[0], p[1], p[2] }, { nx, ny, nz } });
        }
    };
    for (uint32_t i = 0; i < segments; ++i) {
        const float angle = 2.0f * pi * float(i) / float(segments);
        const float next_angle = 2.0f * pi * float((i + 1) % segments) / float(segments);
        const std::array<float, 3> current = { std::cos(angle), -1.0f, std::sin(angle) };
        const std::array<float, 3> next = { std::cos(next_angle), -1.0f, std::sin(next_angle) };
        addTriangle({ 0.0f, 1.0f, 0.0f }, next, current);
        addTriangle({ 0.0f, -1.0f, 0.0f }, current, next);
    }
    index_count = uint32_t(indices.size());
    if (!createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||
        !createBuffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) return false;
    return upload(vertex_buffer, vertices.data(), vertices.size() * sizeof(Vertex)) &&
           upload(index_buffer, indices.data(), indices.size() * sizeof(uint32_t));
}

bool createUniforms() {
    const auto device = graphics::internal::context.device;
    if (!createBuffer(sizeof(Uniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, uniform_buffer)) return false;
    const VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    const VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    if (!check(vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &descriptor_layout),
            "vkCreateDescriptorSetLayout")) return false;
    const VkDescriptorPoolSize pool_size = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1 };
    const VkDescriptorPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
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
    const VkWriteDescriptorSet write = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .pBufferInfo = &buffer_info,
    };
    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
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
    };
    const VkPipelineVertexInputStateCreateInfo vertex_input = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attributes,
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
    position[0] = position[1] = position[2] = 0.0f;
    rotation[0] = 0.35f;
    rotation[1] = rotation[2] = 0.0f;
    scale = 1.0f;
    use_perspective = true;
    camera = Camera{};
    lighting = Lighting{};
    material = Material{};
}

} // namespace

bool initialize(const std::filesystem::path& shader_directory) {
    if (createCone() && createUniforms() && createPipeline(shader_directory)) return true;
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
    for (const Buffer* buffer : { &uniform_buffer, &index_buffer, &vertex_buffer })
        if (buffer->handle) vmaDestroyBuffer(context.allocator, buffer->handle, buffer->allocation);
}

void update([[maybe_unused]] double time) {
    const auto& io = ImGui::GetIO();
    const float step = std::min(io.DeltaTime, 0.1f);
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
    ImGui::SetNextWindowPos(ImVec2(15, 15), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(420, std::min(665.0f, io.DisplaySize.y - 30.0f)), ImGuiCond_Always);
    ImGui::Begin("Lab 2 - Camera and lighting", nullptr, ImGuiWindowFlags_NoResize);
    ImGui::TextUnformatted("Variant 9: cone | Blinn-Phong | Grade 3");
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
        ImGui::ColorEdit3("Albedo", material.albedo);
        ImGui::ColorEdit3("Specular color", material.specular);
        ImGui::SliderFloat("Shininess", &material.shininess, 1.0f, 256.0f, "%.1f", clamp);
    }
    if (ImGui::CollapsingHeader("Model (Lab 1)")) {
        ImGui::DragFloat3("Translation", position, 0.02f, -3.0f, 3.0f, "%.2f", clamp);
        ImGui::SliderAngle("Rotation X", &rotation[0], -180.0f, 180.0f, "%.1f deg", clamp);
        ImGui::SliderAngle("Rotation Y", &rotation[1], -180.0f, 180.0f, "%.1f deg", clamp);
        ImGui::SliderAngle("Rotation Z", &rotation[2], -180.0f, 180.0f, "%.1f deg", clamp);
        ImGui::SliderFloat("Scale", &scale, 0.1f, 3.0f, "%.2f", clamp);
        ImGui::TextUnformatted("Arrows: rotate X/Y | Z/C: rotate Z");
        ImGui::TextUnformatted("W/A/S/D: move X/Y | Q/E: move Z | +/-: scale");
    }
    ImGui::PopItemWidth();
    ImGui::Text("%u triangles | %.0f FPS", index_count / 3, io.Framerate);
    ImGui::End();
}

void render(const graphics::internal::FrameData& fd) {
    const auto& context = graphics::internal::context;
    const float aspect = float(context.swapchain_extent.width) / float(context.swapchain_extent.height);
    const Mat4 model = translation(position[0], position[1], position[2]) *
        rotationZ(rotation[2]) * rotationY(rotation[1]) * rotationX(rotation[0]) * scaling(scale);
    const float radians = pi / 180.0f;
    const Mat4 view = viewMatrix(camera.position[0], camera.position[1], camera.position[2],
        camera.rotation[0] * radians, camera.rotation[1] * radians, camera.rotation[2] * radians);
    const Mat4 projection = use_perspective ?
        perspective(camera.fov * radians, aspect, camera.near_plane, camera.far_plane) :
        orthographic(camera.ortho_half_height, aspect, camera.near_plane, camera.far_plane);
    const float azimuth = lighting.azimuth * radians;
    const float elevation = lighting.elevation * radians;
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
        // Orthographic viewing rays are parallel to the camera's local +Z axis.
        { view.at(2, 0), view.at(2, 1), view.at(2, 2), 0.0f },
    };
    // prepare() has already waited for the previous frame before this UBO write.
    upload(uniform_buffer, &uniforms, sizeof(uniforms));

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
    vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
    vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
    vkCmdEndRenderPass(fd.command_buffer);
    vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application
