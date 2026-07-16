#include "geometry/CacheFileManager.h"
#include "core/Logger.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstring>
#include <format>

namespace geometry {

    namespace {

        // Wraps VirtualAlloc/VirtualFree so a 4KB-page-aligned scratch buffer is always released,
        // even on an early-return error path. VirtualAlloc guarantees an allocation address
        // aligned to the OS page granularity (4096 bytes on Windows x64), which satisfies
        // FILE_FLAG_NO_BUFFERING's sector-alignment requirement for any real storage device (512
        // or 4096-byte sectors).
        class AlignedPageBuffer {
        public:
            explicit AlignedPageBuffer(size_t sizeBytes) : m_SizeBytes(sizeBytes) {
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
            size_t Size() const { return m_SizeBytes; }
            bool IsValid() const { return m_Data != nullptr; }

        private:
            uint8_t* m_Data = nullptr;
            size_t m_SizeBytes = 0;
        };

        // 4096-byte-aligned, heap-allocated (via std::make_shared, which correctly honors
        // over-alignment as of C++17) page buffer for the asynchronous read path, where the
        // buffer must outlive the initiating call and be shared into the future's continuation.
        struct alignas(4096) AlignedPage {
            uint8_t bytes[kPageSizeBytes];
        };
        static_assert(sizeof(AlignedPage) == kPageSizeBytes, "AlignedPage must be exactly one page");

        uint64_t RoundUpToPage(uint64_t sizeBytes) {
            return ((sizeBytes + kPageSizeBytes - 1u) / kPageSizeBytes) * kPageSizeBytes;
        }

        // Writes `dataSizeBytes` bytes from `data`, followed by explicit zero bytes, so exactly
        // `paddedSizeBytes` bytes land in the file -- the "remplit les espaces vides (padding)
        // avec des zéros" requirement, made visible directly in the write path rather than relying
        // on the caller's struct having already been zero-initialized. `paddedSizeBytes` must be a
        // multiple of kPageSizeBytes and >= dataSizeBytes.
        bool WriteZeroPaddedSection(HANDLE fileHandle, const void* data, size_t dataSizeBytes, size_t paddedSizeBytes) {
            AlignedPageBuffer scratch(paddedSizeBytes);
            if (!scratch.IsValid()) {
                LOG_ERROR(std::format(
                    "[CacheFileManager] VirtualAlloc failed for a {}-byte zero-padded write scratch buffer!", paddedSizeBytes));
                return false;
            }

            std::memset(scratch.Get(), 0, paddedSizeBytes);           // Explicit zero-fill of the whole section...
            if (dataSizeBytes > 0) {
                std::memcpy(scratch.Get(), data, dataSizeBytes);      // ...then overwrite the front with real data.
            }

            DWORD bytesWritten = 0;
            BOOL ok = WriteFile(fileHandle, scratch.Get(), static_cast<DWORD>(paddedSizeBytes), &bytesWritten, nullptr);
            return ok != FALSE && bytesWritten == static_cast<DWORD>(paddedSizeBytes);
        }

    } // namespace

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
                    LOG_WARNING(std::format(
                        "[CacheFileManager] Failed to delete stale cache file '{}': {}",
                        entry.path().string(), removeEc.message()));
                }
            }
        }

        LOG_INFO(std::format(
            "[CacheFileManager] Purged {} stale .cache file(s) from '{}'.",
            deletedCount, directory.string()));
    }

    bool CacheFileManager::WriteCacheFile(
        const std::filesystem::path& filePath,
        std::vector<ClusterIndexEntry>& indexEntries,
        const std::vector<DAGNodeEntry>& dagEntries,
        const std::vector<ClusterData>& clusterData,
        uint32_t entityCount) const {

        const uint32_t clusterCount = static_cast<uint32_t>(indexEntries.size());
        if (dagEntries.size() != clusterCount || clusterData.size() != clusterCount) {
            LOG_ERROR(std::format(
                "[CacheFileManager] WriteCacheFile: mismatched section sizes (index={}, dag={}, data={})",
                indexEntries.size(), dagEntries.size(), clusterData.size()));
            return false;
        }

        // --- Compute the on-disk layout up front: every offset is a multiple of kPageSizeBytes ---
        const uint64_t clusterIndexTableOffset = kCacheFileHeaderPaddedSizeBytes;
        const uint64_t clusterIndexTableSizeBytes = static_cast<uint64_t>(clusterCount) * sizeof(ClusterIndexEntry);
        const uint64_t clusterIndexTablePaddedSizeBytes = RoundUpToPage(clusterIndexTableSizeBytes);

        const uint64_t dagTableOffset = clusterIndexTableOffset + clusterIndexTablePaddedSizeBytes;
        const uint64_t dagTableSizeBytes = static_cast<uint64_t>(clusterCount) * sizeof(DAGNodeEntry);
        const uint64_t dagTablePaddedSizeBytes = RoundUpToPage(dagTableSizeBytes);

        const uint64_t geometryDataBaseOffset = dagTableOffset + dagTablePaddedSizeBytes;
        const uint64_t totalFileSizeBytes = geometryDataBaseOffset + static_cast<uint64_t>(clusterCount) * kPageSizeBytes;

        // Now that the layout is known, stamp each cluster's own page-aligned block address back
        // into its index entry -- the writer, not the caller, owns this decision (see the header
        // doc comment on WriteCacheFile).
        for (uint32_t i = 0; i < clusterCount; ++i) {
            indexEntries[i].virtualAddress = geometryDataBaseOffset + static_cast<uint64_t>(i) * kPageSizeBytes;
            indexEntries[i].blockSizeBytes = kPageSizeBytes;
        }

        CacheFileHeader header{};
        header.magic = CacheFileHeader::kMagic;
        header.version = CacheFileHeader::kVersion;
        header.clusterCount = clusterCount;
        header.entityCount = entityCount;
        header.clusterIndexTableOffset = clusterIndexTableOffset;
        header.clusterIndexTableSizeBytes = clusterIndexTableSizeBytes;
        header.dagTableOffset = dagTableOffset;
        header.dagTableSizeBytes = dagTableSizeBytes;
        header.geometryDataBaseOffset = geometryDataBaseOffset;
        header.totalFileSizeBytes = totalFileSizeBytes;

        HANDLE fileHandle = CreateFileW(
            filePath.wstring().c_str(),
            GENERIC_WRITE,
            0, // No sharing while writing.
            nullptr,
            CREATE_ALWAYS,
            FILE_ATTRIBUTE_NORMAL,
            nullptr);

        if (fileHandle == INVALID_HANDLE_VALUE) {
            LOG_ERROR(std::format(
                "[CacheFileManager] CreateFileW failed for '{}' (GetLastError={})",
                filePath.string(), GetLastError()));
            return false;
        }

        // --- Section 1: header, zero-padded to kCacheFileHeaderPaddedSizeBytes -----------------
        if (!WriteZeroPaddedSection(fileHandle, &header, sizeof(CacheFileHeader), kCacheFileHeaderPaddedSizeBytes)) {
            LOG_ERROR(std::format("[CacheFileManager] Failed writing the header to '{}' (GetLastError={})", filePath.string(), GetLastError()));
            CloseHandle(fileHandle);
            return false;
        }

        // --- Section 2: cluster index table, zero-padded to the next page boundary -------------
        if (!WriteZeroPaddedSection(fileHandle, indexEntries.data(), static_cast<size_t>(clusterIndexTableSizeBytes), static_cast<size_t>(clusterIndexTablePaddedSizeBytes))) {
            LOG_ERROR(std::format("[CacheFileManager] Failed writing the cluster index table to '{}' (GetLastError={})", filePath.string(), GetLastError()));
            CloseHandle(fileHandle);
            return false;
        }

        // --- Section 3: DAG table, zero-padded to the next page boundary -----------------------
        if (!WriteZeroPaddedSection(fileHandle, dagEntries.data(), static_cast<size_t>(dagTableSizeBytes), static_cast<size_t>(dagTablePaddedSizeBytes))) {
            LOG_ERROR(std::format("[CacheFileManager] Failed writing the DAG table to '{}' (GetLastError={})", filePath.string(), GetLastError()));
            CloseHandle(fileHandle);
            return false;
        }

        // --- Section 4: one page-aligned, zero-padded ClusterData block per cluster ------------
        for (uint32_t i = 0; i < clusterCount; ++i) {
            if (!WriteZeroPaddedSection(fileHandle, &clusterData[i], sizeof(ClusterData), kPageSizeBytes)) {
                LOG_ERROR(std::format(
                    "[CacheFileManager] Failed writing cluster {}'s geometry block to '{}' (GetLastError={})",
                    i, filePath.string(), GetLastError()));
                CloseHandle(fileHandle);
                return false;
            }
        }

        CloseHandle(fileHandle);

        LOG_INFO(std::format(
            "[CacheFileManager] Wrote '{}': {} cluster(s) across {} entit(y/ies), {} bytes total "
            "(index table {} B, DAG table {} B, geometry {} B).",
            filePath.string(), clusterCount, entityCount, totalFileSizeBytes,
            clusterIndexTablePaddedSizeBytes, dagTablePaddedSizeBytes, static_cast<uint64_t>(clusterCount) * kPageSizeBytes));
        return true;
    }

    bool CacheFileManager::ReadHeader(const std::filesystem::path& filePath, CacheFileHeader& outHeader) const {
        HANDLE fileHandle = CreateFileW(
            filePath.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fileHandle == INVALID_HANDLE_VALUE) {
            LOG_ERROR(std::format(
                "[CacheFileManager] CreateFileW (read header) failed for '{}' (GetLastError={})", filePath.string(), GetLastError()));
            return false;
        }

        DWORD bytesRead = 0;
        BOOL ok = ReadFile(fileHandle, &outHeader, static_cast<DWORD>(sizeof(CacheFileHeader)), &bytesRead, nullptr);
        CloseHandle(fileHandle);

        if (ok == FALSE || bytesRead != sizeof(CacheFileHeader)) {
            LOG_ERROR(std::format("[CacheFileManager] Failed reading the header from '{}'", filePath.string()));
            return false;
        }
        if (outHeader.magic != CacheFileHeader::kMagic || outHeader.version != CacheFileHeader::kVersion) {
            LOG_ERROR(std::format(
                "[CacheFileManager] '{}' has an unrecognized magic/version (magic=0x{:08X}, version={})",
                filePath.string(), outHeader.magic, outHeader.version));
            return false;
        }
        return true;
    }

    namespace {
        // Shared implementation for ReadClusterIndexTable/ReadDAGTable: reads `entryCount`
        // fixed-size entries starting at `tableOffset`.
        template <typename EntryType>
        bool ReadTable(const std::filesystem::path& filePath, uint64_t tableOffset, uint32_t entryCount, std::vector<EntryType>& outEntries) {
            outEntries.assign(entryCount, EntryType{});
            if (entryCount == 0) {
                return true;
            }

            HANDLE fileHandle = CreateFileW(
                filePath.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (fileHandle == INVALID_HANDLE_VALUE) {
                LOG_ERROR(std::format(
                    "[CacheFileManager] CreateFileW (read table) failed for '{}' (GetLastError={})", filePath.string(), GetLastError()));
                return false;
            }

            LARGE_INTEGER seekOffset{};
            seekOffset.QuadPart = static_cast<LONGLONG>(tableOffset);
            if (!SetFilePointerEx(fileHandle, seekOffset, nullptr, FILE_BEGIN)) {
                LOG_ERROR(std::format("[CacheFileManager] SetFilePointerEx failed for '{}'", filePath.string()));
                CloseHandle(fileHandle);
                return false;
            }

            DWORD bytesToRead = static_cast<DWORD>(entryCount * sizeof(EntryType));
            DWORD bytesRead = 0;
            BOOL ok = ReadFile(fileHandle, outEntries.data(), bytesToRead, &bytesRead, nullptr);
            CloseHandle(fileHandle);

            if (ok == FALSE || bytesRead != bytesToRead) {
                LOG_ERROR(std::format("[CacheFileManager] Failed reading a table from '{}'", filePath.string()));
                return false;
            }
            return true;
        }
    }

    bool CacheFileManager::ReadClusterIndexTable(
        const std::filesystem::path& filePath, const CacheFileHeader& header, std::vector<ClusterIndexEntry>& outEntries) const {
        return ReadTable(filePath, header.clusterIndexTableOffset, header.clusterCount, outEntries);
    }

    bool CacheFileManager::ReadDAGTable(
        const std::filesystem::path& filePath, const CacheFileHeader& header, std::vector<DAGNodeEntry>& outEntries) const {
        return ReadTable(filePath, header.dagTableOffset, header.clusterCount, outEntries);
    }

    std::future<bool> CacheFileManager::ReadClusterDataAsync(
        const std::filesystem::path& filePath, uint64_t virtualAddress, ClusterData& outData) const {

        HANDLE fileHandle = CreateFileW(
            filePath.wstring().c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
            nullptr);

        if (fileHandle == INVALID_HANDLE_VALUE) {
            LOG_ERROR(std::format(
                "[CacheFileManager] CreateFileW (read cluster) failed for '{}' (GetLastError={})",
                filePath.string(), GetLastError()));
            std::promise<bool> failedPromise;
            failedPromise.set_value(false);
            return failedPromise.get_future();
        }

        HANDLE completionEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (completionEvent == nullptr) {
            LOG_ERROR("[CacheFileManager] CreateEventW failed for overlapped cluster read!");
            CloseHandle(fileHandle);
            std::promise<bool> failedPromise;
            failedPromise.set_value(false);
            return failedPromise.get_future();
        }

        auto page = std::make_shared<AlignedPage>();
        auto overlapped = std::make_shared<OVERLAPPED>();
        ZeroMemory(overlapped.get(), sizeof(OVERLAPPED));
        overlapped->Offset = static_cast<DWORD>(virtualAddress & 0xFFFFFFFFu);
        overlapped->OffsetHigh = static_cast<DWORD>((virtualAddress >> 32) & 0xFFFFFFFFu);
        overlapped->hEvent = completionEvent;

        BOOL immediateResult = ReadFile(fileHandle, page->bytes, static_cast<DWORD>(kPageSizeBytes), nullptr, overlapped.get());
        DWORD lastError = GetLastError();

        if (immediateResult == FALSE && lastError != ERROR_IO_PENDING) {
            LOG_ERROR(std::format(
                "[CacheFileManager] ReadFile failed for '{}' at offset {} (GetLastError={})",
                filePath.string(), virtualAddress, lastError));
            CloseHandle(completionEvent);
            CloseHandle(fileHandle);
            std::promise<bool> failedPromise;
            failedPromise.set_value(false);
            return failedPromise.get_future();
        }

        // The wait-for-completion + result validation + final copy-out runs off the calling
        // thread on a pooled async task, which is what makes this call asynchronous from the
        // caller's perspective; the OS-level I/O itself is already asynchronous by virtue of
        // FILE_FLAG_OVERLAPPED.
        return std::async(std::launch::async, [fileHandle, completionEvent, overlapped, page, &outData]() -> bool {
            DWORD bytesTransferred = 0;
            BOOL ok = GetOverlappedResult(fileHandle, overlapped.get(), &bytesTransferred, TRUE /* block until done */);
            DWORD waitError = GetLastError();

            CloseHandle(completionEvent);
            CloseHandle(fileHandle);

            if (ok == FALSE || bytesTransferred != kPageSizeBytes) {
                LOG_ERROR(std::format(
                    "[CacheFileManager] Overlapped cluster read did not complete cleanly "
                    "(GetLastError={}, bytesTransferred={}).",
                    waitError, bytesTransferred));
                return false;
            }
            std::memcpy(&outData, page->bytes, sizeof(ClusterData));
            return true;
            });
    }

}
