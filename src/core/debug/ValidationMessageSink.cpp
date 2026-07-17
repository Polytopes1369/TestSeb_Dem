#ifndef NDEBUG

#include "core/debug/ValidationMessageSink.h"

namespace debugpipeline {

    std::mutex ValidationMessageSink::s_Mutex;
    std::vector<ValidationMessage> ValidationMessageSink::s_Messages;

    void ValidationMessageSink::Push(uint32_t severity, const std::string& text) {
        std::lock_guard<std::mutex> lock(s_Mutex);
        s_Messages.push_back(ValidationMessage{ severity, text });
    }

    void ValidationMessageSink::BeginWindow() {
        std::lock_guard<std::mutex> lock(s_Mutex);
        s_Messages.clear();
    }

    std::vector<ValidationMessage> ValidationMessageSink::EndWindow() {
        std::lock_guard<std::mutex> lock(s_Mutex);
        std::vector<ValidationMessage> result = std::move(s_Messages);
        s_Messages.clear();
        return result;
    }

}

#endif // NDEBUG
