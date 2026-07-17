#include "io/VirtualTextureCacheFileManager.h"
#include "core/Logger.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstring>
#include <format>

namespace io {

    namespace {

        // Identical role/implementation to geometry::CacheFileManager.cpp's own anonymous-namespace
        // AlignedPageBuffer -- see that file's comment for the full VirtualAlloc-alignment rationale.
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
            bool IsValid() const { return m_Data != nullptr; }

        private:
            uint8_t* m_Data = nullptr;
            size_t m_SizeBytes = 0;
        };

        uint64_t RoundUpToPage(uint64_t sizeBytes) {
            return ((sizeBytes + kVTPageSizeBytes - 1u) / kVTPageSizeBytes) * kVTPageSizeBytes;
        }

        // Identical role to geometry::CacheFileManager.cpp's own WriteZeroPaddedSection -- see that
        // function's comment for why the zero-fill is made explicit in the write path itself.
        bool WriteZeroPaddedSection(HANDLE fileHandle, const void* data, size_t dataSizeBytes, size_t paddedSizeBytes) {
            if (paddedSizeBytes == 0) {
                return true;
            }
            AlignedPageBuffer scratch(paddedSizeBytes);
            if (!scratch.IsValid()) {
                LOG_ERROR(std::format(
                    "[VirtualTextureCacheFileManager] VirtualAlloc failed for a {}-byte zero-padded write scratch buffer!", paddedSizeBytes));
                return false;
            }

            std::memset(scratch.Get(), 0, paddedSizeBytes);
            if (dataSizeBytes > 0) {
                std::memcpy(scratch.Get(), data, dataSizeBytes);
            }

            DWORD bytesWritten = 0;
            BOOL ok = WriteFile(fileHandle, scratch.Get(), static_cast<DWORD>(paddedSizeBytes), &bytesWritten, nullptr);
            return ok != FALSE && bytesWritten == static_cast<DWORD>(paddedSizeBytes);
        }

    } // namespace

    void VirtualTextureCacheFileManager::PurgeExistingCacheFiles(const std::filesystem::path& directory) const {
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
            if (entry.path().extension() == ".vtcache") {
                std::error_code removeEc;
                if (std::filesystem::remove(entry.path(), removeEc)) {
                    ++deletedCount;
                } else {
                    LOG_WARNING(std::format(
                        "[VirtualTextureCacheFileManager] Failed to delete stale cache file '{}': {}",
                        entry.path().string(), removeEc.message()));
                }
            }
        }

        LOG_INFO(std::format(
            "[VirtualTextureCacheFileManager] Purged {} stale .vtcache file(s) from '{}'.",
            deletedCount, directory.string()));
    }

    bool VirtualTextureCacheFileManager::WriteCacheFile(
        const std::filesystem::path& filePath,
        std::vector<VirtualTextureTileIndexEntry>& indexEntries,
        std::vector<VirtualTextureTileData>& tileData,
        uint32_t poolCount, uint32_t tileSizeTexels, uint32_t borderSizeTexels) const {

        const uint32_t tileCount = static_cast<uint32_t>(indexEntries.size());
        if (tileData.size() != tileCount) {
            LOG_ERROR(std::format(
                "[VirtualTextureCacheFileManager] WriteCacheFile: mismatched section sizes (index={}, data={})",
                indexEntries.size(), tileData.size()));
            return false;
        }

        HANDLE fileHandle = CreateFileW(
            filePath.wstring().c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fileHandle == INVALID_HANDLE_VALUE) {
            LOG_ERROR(std::format(
                "[VirtualTextureCacheFileManager] CreateFileW (write) failed for '{}' (GetLastError={})",
                filePath.string(), GetLastError()));
            return false;
        }

        // --- Layout: header (1 page) + tile index table (page-aligned) + per-tile texel blobs
        // (each individually page-aligned) -- see VirtualTextureCacheFormat.h's own header comment. ---
        uint64_t tileIndexTableOffset = kVTCacheFileHeaderPaddedSizeBytes;
        uint64_t tileIndexTableSizeBytes = static_cast<uint64_t>(tileCount) * sizeof(VirtualTextureTileIndexEntry);
        uint64_t tileIndexTablePaddedSizeBytes = RoundUpToPage(tileIndexTableSizeBytes);
        uint64_t tileDataBaseOffset = tileIndexTableOffset + tileIndexTablePaddedSizeBytes;

        // Assign each tile's virtualAddress/blockSizeBytes IN PLACE before writing the index table
        // -- the writer owns physical layout, exactly like geometry::CacheFileManager::WriteCacheFile.
        uint64_t runningOffset = tileDataBaseOffset;
        for (uint32_t i = 0; i < tileCount; ++i) {
            uint64_t blockSizeBytes = RoundUpToPage(static_cast<uint64_t>(tileData[i].texelBytes.size()));
            indexEntries[i].pageKey = tileData[i].pageKey;
            indexEntries[i].virtualAddress = runningOffset;
            indexEntries[i].blockSizeBytes = static_cast<uint32_t>(blockSizeBytes);
            runningOffset += blockSizeBytes;
        }
        uint64_t totalFileSizeBytes = runningOffset;

        // --- Section 0: header ---
        VirtualTextureCacheFileHeader header{};
        header.magic = VirtualTextureCacheFileHeader::kMagic;
        header.version = VirtualTextureCacheFileHeader::kVersion;
        header.tileCount = tileCount;
        header.poolCount = poolCount;
        header.tileSizeTexels = tileSizeTexels;
        header.borderSizeTexels = borderSizeTexels;
        header.tileIndexTableOffset = tileIndexTableOffset;
        header.tileIndexTableSizeBytes = tileIndexTableSizeBytes;
        header.tileDataBaseOffset = tileDataBaseOffset;
        header.totalFileSizeBytes = totalFileSizeBytes;

        if (!WriteZeroPaddedSection(fileHandle, &header, sizeof(header), kVTCacheFileHeaderPaddedSizeBytes)) {
            LOG_ERROR(std::format("[VirtualTextureCacheFileManager] Failed writing header to '{}'", filePath.string()));
            CloseHandle(fileHandle);
            return false;
        }

        // --- Section 1: tile index table ---
        if (!WriteZeroPaddedSection(fileHandle, indexEntries.data(),
            static_cast<size_t>(tileIndexTableSizeBytes), static_cast<size_t>(tileIndexTablePaddedSizeBytes))) {
            LOG_ERROR(std::format("[VirtualTextureCacheFileManager] Failed writing tile index table to '{}'", filePath.string()));
            CloseHandle(fileHandle);
            return false;
        }

        // --- Section 2: per-tile texel blobs, each individually page-aligned (paged in on demand
        // later, exactly like ClusterFormat.h's own ClusterData blocks). ---
        for (uint32_t i = 0; i < tileCount; ++i) {
            uint64_t paddedSizeBytes = RoundUpToPage(static_cast<uint64_t>(tileData[i].texelBytes.size()));
            if (!WriteZeroPaddedSection(fileHandle, tileData[i].texelBytes.data(),
                tileData[i].texelBytes.size(), static_cast<size_t>(paddedSizeBytes))) {
                LOG_ERROR(std::format(
                    "[VirtualTextureCacheFileManager] Failed writing tile data (pageKey={}) to '{}'",
                    indexEntries[i].pageKey, filePath.string()));
                CloseHandle(fileHandle);
                return false;
            }
        }

        CloseHandle(fileHandle);

        LOG_INFO(std::format(
            "[VirtualTextureCacheFileManager] Wrote '{}': {} tile(s), {} pool channel(s)/tile, {} bytes total.",
            filePath.string(), tileCount, poolCount, totalFileSizeBytes));
        return true;
    }

    bool VirtualTextureCacheFileManager::ReadHeader(
        const std::filesystem::path& filePath, VirtualTextureCacheFileHeader& outHeader) const {
        HANDLE fileHandle = CreateFileW(
            filePath.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fileHandle == INVALID_HANDLE_VALUE) {
            LOG_ERROR(std::format(
                "[VirtualTextureCacheFileManager] CreateFileW (read header) failed for '{}' (GetLastError={})",
                filePath.string(), GetLastError()));
            return false;
        }

        DWORD bytesRead = 0;
        BOOL ok = ReadFile(fileHandle, &outHeader, static_cast<DWORD>(sizeof(VirtualTextureCacheFileHeader)), &bytesRead, nullptr);
        CloseHandle(fileHandle);

        if (ok == FALSE || bytesRead != sizeof(VirtualTextureCacheFileHeader)) {
            LOG_ERROR(std::format("[VirtualTextureCacheFileManager] Failed reading the header from '{}'", filePath.string()));
            return false;
        }
        if (outHeader.magic != VirtualTextureCacheFileHeader::kMagic || outHeader.version != VirtualTextureCacheFileHeader::kVersion) {
            LOG_ERROR(std::format(
                "[VirtualTextureCacheFileManager] '{}' has an unrecognized magic/version (magic=0x{:08X}, version={})",
                filePath.string(), outHeader.magic, outHeader.version));
            return false;
        }
        return true;
    }

    bool VirtualTextureCacheFileManager::ReadTileIndexTable(
        const std::filesystem::path& filePath, const VirtualTextureCacheFileHeader& header,
        std::vector<VirtualTextureTileIndexEntry>& outEntries) const {
        HANDLE fileHandle = CreateFileW(
            filePath.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (fileHandle == INVALID_HANDLE_VALUE) {
            LOG_ERROR(std::format(
                "[VirtualTextureCacheFileManager] CreateFileW (read tile index table) failed for '{}' (GetLastError={})",
                filePath.string(), GetLastError()));
            return false;
        }

        LARGE_INTEGER offset{};
        offset.QuadPart = static_cast<LONGLONG>(header.tileIndexTableOffset);
        if (!SetFilePointerEx(fileHandle, offset, nullptr, FILE_BEGIN)) {
            LOG_ERROR(std::format("[VirtualTextureCacheFileManager] SetFilePointerEx failed for '{}'", filePath.string()));
            CloseHandle(fileHandle);
            return false;
        }

        outEntries.resize(header.tileCount);
        size_t expectedBytes = static_cast<size_t>(header.tileCount) * sizeof(VirtualTextureTileIndexEntry);
        if (expectedBytes != header.tileIndexTableSizeBytes) {
            LOG_ERROR(std::format(
                "[VirtualTextureCacheFileManager] Tile index table size mismatch in '{}': header says {} bytes, {} entries expect {} bytes",
                filePath.string(), header.tileIndexTableSizeBytes, header.tileCount, expectedBytes));
            CloseHandle(fileHandle);
            return false;
        }

        DWORD bytesRead = 0;
        BOOL ok = expectedBytes == 0 ? TRUE
            : ReadFile(fileHandle, outEntries.data(), static_cast<DWORD>(expectedBytes), &bytesRead, nullptr);
        CloseHandle(fileHandle);

        if (ok == FALSE || (expectedBytes > 0 && bytesRead != expectedBytes)) {
            LOG_ERROR(std::format("[VirtualTextureCacheFileManager] Failed reading the tile index table from '{}'", filePath.string()));
            return false;
        }
        return true;
    }

}
