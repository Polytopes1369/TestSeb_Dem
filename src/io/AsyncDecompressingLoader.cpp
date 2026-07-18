#include "io/AsyncDecompressingLoader.h"
#include "io/BlockCodec.h"
#include "core/Logger.h"

#include <format>
#include <utility>

namespace io {

    namespace {

        // Holds one in-flight request's decompression result between the core::LoadingManager
        // background job (which writes it) and its completion callback (which reads it) -- safe
        // without extra locking because LoadingManager guarantees onMainThreadComplete is only
        // ever invoked AFTER backgroundWork has returned (see LoadingManager.h's own threading
        // contract), giving the write-then-read here a happens-before relationship for free.
        struct DecompressResult {
            bool success = false;
            std::unique_ptr<uint8_t[]> data;
            uint32_t sizeBytes = 0;
        };

    } // namespace

    AsyncDecompressingLoader::AsyncDecompressingLoader(core::LoadingManager& workerPool, uint32_t ioWorkerThreadCount)
        : m_Streamer(ioWorkerThreadCount), m_WorkerPool(workerPool) {
    }

    AsyncDecompressingLoader::~AsyncDecompressingLoader() {
        Close();
    }

    bool AsyncDecompressingLoader::Open(const std::filesystem::path& filePath) {
        return m_Streamer.Open(filePath);
    }

    void AsyncDecompressingLoader::Close() {
        m_Streamer.Close();
    }

    bool AsyncDecompressingLoader::IsOpen() const {
        return m_Streamer.IsOpen();
    }

    bool AsyncDecompressingLoader::SubmitChunkLoad(const ChunkLoadRequest& request, ChunkLoadCallback onComplete) {
        if (!m_Streamer.IsOpen()) return false;

        void* rawBuffer = geometry::AsyncFileStreamer::AllocateAlignedBuffer(request.alignedReadSizeBytes);
        if (!rawBuffer) return false;

        // shared_ptr (not unique_ptr) with a custom deleter calling FreeAlignedBuffer: this buffer
        // needs to be captured by copy into two nested std::function layers below (the
        // AsyncFileStreamer::ReadCallback, and inside it a core::LoadingManager job) -- std::function
        // requires its captures to be copy-constructible, which unique_ptr is not.
        std::shared_ptr<uint8_t> alignedBuffer(static_cast<uint8_t*>(rawBuffer),
            [](uint8_t* p) { geometry::AsyncFileStreamer::FreeAlignedBuffer(p); });

        auto callback = std::make_shared<ChunkLoadCallback>(std::move(onComplete));
        ChunkLoadRequest req = request;

        // Captures `this` (NOT a copy of the m_WorkerPool reference -- capturing a reference-typed
        // local by reference would capture a reference to that local variable itself, which goes
        // out of scope when SubmitChunkLoad returns, long before this lambda actually runs) so
        // m_WorkerPool is reached indirectly through the still-alive AsyncDecompressingLoader
        // object. Safe because ~AsyncDecompressingLoader calls Close(), which (via
        // AsyncFileStreamer::Close()'s own documented contract) blocks until every in-flight read's
        // callback -- this lambda -- has already run, so `this` can never dangle while this lambda
        // is still pending.
        bool launched = m_Streamer.SubmitRead(req.fileOffset, alignedBuffer.get(), req.alignedReadSizeBytes,
            [this, alignedBuffer, req, callback](bool readSuccess, uint32_t bytesTransferred) {
                // Runs on an AsyncFileStreamer IOCP worker thread -- must stay minimal (see this
                // file's header comment on why decompression must NOT happen here).
                if (!readSuccess || bytesTransferred < req.compressedSizeBytes) {
                    LOG_WARNING(std::format(
                        "[AsyncDecompressingLoader] Chunk read failed or short at offset {}: success={}, bytesTransferred={}, expected={}.",
                        req.fileOffset, readSuccess, bytesTransferred, req.compressedSizeBytes));
                    // Still routed through m_WorkerPool so the caller's callback is ALWAYS invoked
                    // from the same place (a PumpCompletions() drain), success or failure alike --
                    // never invoked directly from this IOCP thread.
                    m_WorkerPool.Submit([] {}, [callback] { if (*callback) (*callback)(false, nullptr, 0); });
                    return;
                }

                auto result = std::make_shared<DecompressResult>();
                m_WorkerPool.Submit(
                    [alignedBuffer, req, result] {
                        // Runs on a core::LoadingManager worker thread (never the IOCP thread, and
                        // never the main thread) -- the actual CPU-bound decompression work.
                        auto decompressed = std::make_unique<uint8_t[]>(req.decompressedSizeBytes);
                        size_t written = DecompressBlock(alignedBuffer.get(), req.compressedSizeBytes, decompressed.get(), req.decompressedSizeBytes);
                        if (written == req.decompressedSizeBytes) {
                            result->success = true;
                            result->data = std::move(decompressed);
                            result->sizeBytes = req.decompressedSizeBytes;
                        } else {
                            LOG_ERROR(std::format(
                                "[AsyncDecompressingLoader] Decompression mismatch at offset {}: produced {} bytes, expected {} -- corrupt or truncated block.",
                                req.fileOffset, written, req.decompressedSizeBytes));
                        }
                    },
                    [callback, result] {
                        if (*callback) (*callback)(result->success, std::move(result->data), result->sizeBytes);
                    });
            });

        return launched;
    }

}
