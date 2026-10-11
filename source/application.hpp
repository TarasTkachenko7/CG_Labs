#pragma once

#include "graphics_internal.hpp"
#include <filesystem>

namespace application {

bool initialize(const std::filesystem::path& shader_directory);
void shutdown();

void update(double time);
void render(const graphics::internal::FrameData& fd);

} // namespace application
