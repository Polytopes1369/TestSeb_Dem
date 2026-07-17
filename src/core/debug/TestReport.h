#pragma once
// Debug-only (whole file compiled out in Release, see the #ifndef NDEBUG guard below): aggregates
// every FeatureTestResult produced by DebugTestPipeline::RunAll and writes ONE self-contained
// Markdown report. Deliberately verbose (raw validation-layer text, exact numeric bounds, config
// state, screenshot paths) rather than a bare PASS/FAIL table: the report is meant to be handed to
// an AI assistant that has no access to the running engine, so every FAIL must be diagnosable from
// the report text alone, without needing to reproduce it first.
#ifndef NDEBUG

#include <cstdint>
#include <string>
#include <vector>

#include "core/debug/ValidationMessageSink.h"

namespace debugpipeline {

    enum class TestStatus {
        Pass,
        Warn,
        Fail
    };

    struct FeatureTestResult {
        std::string name;         // e.g. "Global SDF Bake + Ray March View"
        std::string sourceFile;   // e.g. "src/renderer/passes/GlobalSDFPass.cpp" -- points the reader at the implementation.
        TestStatus status = TestStatus::Fail;
        uint32_t framesExecuted = 0;
        std::string expected;         // What a correct run should look like.
        std::string actual;           // What was actually observed (numbers, not just prose).
        std::vector<ValidationMessage> validationMessages; // Every Vulkan validation-layer message seen during this test's frames.
        std::string screenshotPath;   // Relative to the report file (e.g. "screenshots/06_gi_hwrt.bmp"); empty if none captured.
        std::string notes;            // Freeform: caveats, what this test does NOT verify, pointers to demo_log.txt, etc.
    };

    struct TestReportHeader {
        std::string timestampIso8601;
        std::string gitCommitHash;
        std::string gpuName;
        std::string driverVersionText;
        std::string activeConfigPreset; // config::g_ActiveProfileName at the time of the run.
        std::string buildConfig;        // Always "Debug" -- this whole pipeline cannot exist in Release.
    };

    class TestReport {
    public:
        void SetHeader(const TestReportHeader& header) { m_Header = header; }
        void AddResult(FeatureTestResult result) { m_Results.push_back(std::move(result)); }

        // Writes `<outputDir>/report.md` (creating outputDir if needed). Screenshots are expected
        // to already have been written into `<outputDir>/screenshots/` by the caller before this is
        // called. Returns the full path written, or an empty string on failure.
        std::string WriteMarkdown(const std::string& outputDir) const;

        uint32_t FailCount() const;

    private:
        TestReportHeader m_Header;
        std::vector<FeatureTestResult> m_Results;
    };

}

#endif // NDEBUG
