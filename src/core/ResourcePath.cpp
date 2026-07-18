#include "core/ResourcePath.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

namespace core {

    std::filesystem::path ResolveExeRelativePath(const std::filesystem::path& relativePath) {
        if (relativePath.is_absolute()) {
            return relativePath;
        }

        // GetModuleFileNameW's result is fixed for the process lifetime, so it is resolved once
        // and cached rather than re-querying the OS on every call (shader loads call this per file).
        static const std::filesystem::path exeDir = [] {
            wchar_t buffer[MAX_PATH];
            const DWORD length = GetModuleFileNameW(nullptr, buffer, MAX_PATH);
            if (length == 0 || length == MAX_PATH) {
                return std::filesystem::path{};
            }
            return std::filesystem::path(buffer).parent_path();
        }();

        return exeDir.empty() ? relativePath : exeDir / relativePath;
    }

} // namespace core
