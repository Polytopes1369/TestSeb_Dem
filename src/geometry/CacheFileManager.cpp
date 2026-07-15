#include "geometry/CacheFileManager.h"
#include "core/Logger.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstring>
#include <format>

namespace geometry {

    namespace {
        // Wraps VirtualAlloc/VirtualFree so the 4KB-aligned scratch buffer used for the write
        // path is always released, even on an early-return error path. VirtualAlloc guarantees
        // an allocation address aligned to the OS page granularity (4096 bytes on Windows x64),
        // which satisfies FILE_FLAG_NO_BUFFERING's sector-alignment requirement for any real
        // storage device (512 or 4096-byte sectors).
        class AlignedPageBuffer {
        public:
            explicit AlignedPageBuffer(size_t sizeBytes) {
                m_Data = static_cast<uint8_t*>(VirtualAlloc(nullptr, sizeBytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
            }
            ~AlignedPageBuffer() {
                if (m_Data != nullptr) {
                    VirtualFree(m_Data, 0, MEM_RELEASE);
                }
            }

            AlignedPageBuffer(const AlignedPageBuffer&) = delete;
            AlignedPageBuffer& operator=(const AlignedPageBuffer&) = delete;

            uint8_t* Get() const { return m_Data; }
            bool IsValid() const { return m_Data != nullptr; }

        private:
            uint8_t* m_Data = nullptr;
        };
    }

    void CacheFileManager::PurgeExistingCacheFiles(const std::filesystem::path& directory) {
        std::error_code existsEc;
        if (!std::filesystem::exists(directory, existsEc)) {
            return;
        }

        uint32_t deletedCount = 0;
        std::error_code iterateEc;
        for (const auto& entry : std::filesystem::directory_iterator(directory, iterateEc)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            if (entry.path().extension() == ".cache") {
                std::error_code removeEc;
                if (std::filesystem::remove(entry.path(), removeEc)) {
                    ++deletedCount;
                }
                else {
                    Logger::Log(LogLevel::Warning, std::format(
                        "[CacheFileManager] Failed to delete stale cache file '{}': {}",
                        entry.path().string(), removeEc.message()));
                }
            }
        }

        Logger::Log(LogLevel::Info, std::format(
            "[CacheFileManager] Purged {} stale .cache file(s) from '{}'.",
            deletedCount, directory.string()));
    }

    bool CacheFileManager::WriteCacheFile(const std::filesystem::path& filePath,
        const CacheHeader& header,
        const std::vector<Page>& pages) const {
        // Plain buffered write: the on-disk layout only needs to be sector-clean for the
        // unbuffered *read* path (ReadPageAsync). What's required here is that every RAM buffer
        // handed to WriteFile is itself explicitly 4KB-aligned, which the VirtualAlloc-backed
        // AlignedPageBuffer below guarantees regardless of what the CRT heap allocator would do.
        HANDLE fileHandle = CreateFileW(
            filePath.wstring().c_str(),
            GENERIC_WRITE,
            0, // No sharing while writing.
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);

        if (fileHandle == INVALID_HANDLE_VALUE) {
            Logger::Log(LogLevel::Error, std::format(
                "[CacheFileManager] CreateFileW failed for '{}' (GetLastError={})",
                filePath.string(), GetLastError()));
            return false;
        }

        AlignedPageBuffer scratch(CacheHeader::kHeaderSizeBytes);
        if (!scratch.IsValid()) {
            Logger::Log(LogLevel::Error, "[CacheFileManager] VirtualAlloc failed for the 4KB write scratch buffer!");
            CloseHandle(fileHandle);
            return false;
        }

        auto writeAligned = [&](const void* src, DWORD sizeBytes) -> bool {
            std::memcpy(scratch.Get(), src, sizeBytes);
            DWORD bytesWritten = 0;
            BOOL ok = WriteFile(fileHandle, scratch.Get(), sizeBytes, &bytesWritten, nullptr);
            return ok != FALSE && bytesWritten == sizeBytes;
            };

        static_assert(sizeof(CacheHeader) == CacheHeader::kHeaderSizeBytes, "CacheHeader must stay page-sized");
        if (!writeAligned(&header, static_cast<DWORD>(sizeof(CacheHeader)))) {
            Logger::Log(LogLevel::Error, std::format(
                "[CacheFileManager] Failed writing CacheHeader to '{}' (GetLastError={})",
                filePath.string(), GetLastError()));
            CloseHandle(fileHandle);
            return false;
        }

        for (const Page& page : pages) {
            if (!writeAligned(&page, static_cast<DWORD>(sizeof(Page)))) {
                Logger::Log(LogLevel::Error, std::format(
                    "[CacheFileManager] Failed writing a Page to '{}' (GetLastError={})",
                    filePath.string(), GetLastError()));
                CloseHandle(fileHandle);
                return false;
            }
        }

        CloseHandle(fileHandle);

        Logger::Log(LogLevel::Info, std::format(
            "[CacheFileManager] Wrote '{}': 1 CacheHeader + {} Page(s) = {} bytes.",
            filePath.string(), pages.size(), (1ull + pages.size()) * Page::kPageSizeBytes));
        return true;
    }

    std::future<bool> CacheFileManager::ReadPageAsync(const std::filesystem::path& filePath,
        uint32_t pageIndex,
        Page& outPage) const {
        // outPage must already be 4096-byte aligned in memory: Page::alignas(4096) guarantees
        // this for any Page allocated on the stack, via `new`, or as a class member — the
        // compiler enforces the alignment at every allocation site.
        HANDLE fileHandle = CreateFileW(
            filePath.wstring().c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
            nullptr);

        if (fileHandle == INVALID_HANDLE_VALUE) {
            Logger::Log(LogLevel::Error, std::format(
                "[CacheFileManager] CreateFileW (read) failed for '{}' (GetLastError={})",
                filePath.string(), GetLastError()));
            std::promise<bool> failedPromise;
            failedPromise.set_value(false);
            return failedPromise.get_future();
        }

        HANDLE completionEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (completionEvent == nullptr) {
            Logger::Log(LogLevel::Error, "[CacheFileManager] CreateEventW failed for overlapped page read!");
            CloseHandle(fileHandle);
            std::promise<bool> failedPromise;
            failedPromise.set_value(false);
            return failedPromise.get_future();
        }

        // Pages live immediately after the single-page CacheHeader, so file offset =
        // (pageIndex + 1) * Page::kPageSizeBytes.
        uint64_t byteOffset = static_cast<uint64_t>(pageIndex + 1u) * Page::kPageSizeBytes;

        auto overlapped = std::make_shared<OVERLAPPED>();
        ZeroMemory(overlapped.get(), sizeof(OVERLAPPED));
        overlapped->Offset = static_cast<DWORD>(byteOffset & 0xFFFFFFFFu);
        overlapped->OffsetHigh = static_cast<DWORD>((byteOffset >> 32) & 0xFFFFFFFFu);
        overlapped->hEvent = completionEvent;

        BOOL immediateResult = ReadFile(fileHandle, &outPage, static_cast<DWORD>(sizeof(Page)), nullptr, overlapped.get());
        DWORD lastError = GetLastError();

        if (immediateResult == FALSE && lastError != ERROR_IO_PENDING) {
            Logger::Log(LogLevel::Error, std::format(
                "[CacheFileManager] ReadFile failed for '{}' page {} (GetLastError={})",
                filePath.string(), pageIndex, lastError));
            CloseHandle(completionEvent);
            CloseHandle(fileHandle);
            std::promise<bool> failedPromise;
            failedPromise.set_value(false);
            return failedPromise.get_future();
        }

        // The wait-for-completion + result validation runs off the calling thread on a pooled
        // async task, which is what makes this call asynchronous from the caller's perspective;
        // the OS-level I/O itself is already asynchronous by virtue of FILE_FLAG_OVERLAPPED.
        return std::async(std::launch::async, [fileHandle, completionEvent, overlapped, &outPage]() -> bool {
            DWORD bytesTransferred = 0;
            BOOL ok = GetOverlappedResult(fileHandle, overlapped.get(), &bytesTransferred, TRUE /* block until done */);
            DWORD waitError = GetLastError();

            CloseHandle(completionEvent);
            CloseHandle(fileHandle);

            if (ok == FALSE || bytesTransferred != sizeof(Page)) {
                Logger::Log(LogLevel::Error, std::format(
                    "[CacheFileManager] Overlapped page read did not complete cleanly "
                    "(GetLastError={}, bytesTransferred={}).",
                    waitError, bytesTransferred));
                return false;
            }
            (void)outPage; // outPage was filled in-place by the OS; nothing left to copy here.
            return true;
            });
    }

}
