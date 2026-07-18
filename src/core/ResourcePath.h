#pragma once
#include <filesystem>

namespace core {

    // Resolves a resource path (shaders, caches, configs, world data) the engine ships as a
    // relative literal against the running executable's own directory, not the process current
    // working directory -- CWD depends on how DemoSceneVK.exe was launched (build dir, repo root,
    // a script that cd's elsewhere) while every deployed resource always sits next to the .exe.
    // Returns `relativePath` unchanged if it is already absolute.
    std::filesystem::path ResolveExeRelativePath(const std::filesystem::path& relativePath);

} // namespace core
