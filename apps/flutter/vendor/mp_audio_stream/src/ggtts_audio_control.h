#ifndef GGTTS_AUDIO_CONTROL_H
#define GGTTS_AUDIO_CONTROL_H

#include <cstdint>
#include <functional>
#include <mutex>

// Ensures partially initialized device/ring resources are released on every
// failure return. The owner supplies cleanup and commits only after admission.
class GgttsInitGuard {
public:
    explicit GgttsInitGuard(std::function<void()> cleanup) : cleanup_(cleanup) {}
    ~GgttsInitGuard() { if (!committed_) cleanup_(); }
    void commit() { committed_ = true; }
private:
    std::function<void()> cleanup_;
    bool committed_ = false;
};

// All calls that mutate the output device or its software ring are serialized.
// The backend must not call this object from its audio callback.
class GgttsAudioControl {
public:
    enum class State { idle, running, paused, stopped };
    struct Backend {
        virtual ~Backend() = default;
        virtual bool start() = 0;
        virtual bool stop() = 0;
        virtual void clear() = 0;
        virtual void destroy() = 0;
        virtual uint64_t queued() = 0;
    };

    explicit GgttsAudioControl(Backend& backend) : backend_(backend) {}

    bool begin() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != State::idle && state_ != State::stopped) return false;
        state_ = State::running;
        return true;
    }

    bool pause() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != State::running || !backend_.stop()) return false;
        state_ = State::paused;
        return true;
    }

    bool resume() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ != State::paused || !backend_.start()) return false;
        state_ = State::running;
        return true;
    }

    bool stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == State::stopped) return true;
        if (state_ != State::running && state_ != State::paused) return false;
        if (state_ == State::running && !backend_.stop()) return false;
        backend_.clear();
        state_ = State::stopped;
        return true;
    }

    // Always destroy the device even when the explicit stop fails. The
    // result reports that failure; no producer can race device destruction.
    bool teardown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == State::idle) return true;
        const bool stopped = state_ != State::running || backend_.stop();
        backend_.destroy();
        backend_.clear();
        state_ = State::idle;
        return stopped;
    }

    // Empty here means handed off to the device, not heard by the listener.
    bool softwareEmpty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_ == State::running && backend_.queued() == 0;
    }

    template <typename F>
    int64_t push(F operation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == State::paused) return -2;
        if (state_ != State::running) return -3;
        return operation();
    }

    State state() {
        std::lock_guard<std::mutex> lock(mutex_);
        return state_;
    }

private:
    Backend& backend_;
    std::mutex mutex_;
    State state_ = State::idle;
};

// A reservation is one read. IDs never wrap or get reused. The same admission
// lock serializes initialization, controls, teardown and legacy calls.
class GgttsAudioLifecycle {
public:
    std::mutex& mutex() { return mutex_; }
    bool legacyAllowed() const { return current_ == 0; } // caller holds mutex()

    uint64_t reserve() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (current_ != 0 || next_ == 0) return 0;
        current_ = next_++;
        cancelled_ = false;
        initialized_ = false;
        return current_;
    }

    template <typename F>
    int64_t initialize(uint64_t id, F operation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!id || id != current_ || cancelled_ || initialized_) return -1;
        const int64_t result = operation();
        if (result == 0) initialized_ = true;
        return result;
    }

    int64_t stopSession(uint64_t id, GgttsAudioControl& control) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!id || id != current_) return -1;
        if (cancelled_) return 0;
        if (initialized_ && !control.stop()) return -1;
        cancelled_ = true;
        return 0;
    }

    template <typename F>
    int64_t invokeSession(uint64_t id, F operation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!id || id != current_ || cancelled_ || !initialized_) return -1;
        return operation();
    }

    int64_t sessionState(uint64_t id, GgttsAudioControl& control) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!id || id != current_) return -1;
        if (cancelled_) return 3;
        if (!initialized_) return 0;
        return static_cast<int64_t>(control.state());
    }

    template <typename F>
    int64_t pushSession(uint64_t id, F operation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!id || id != current_ || cancelled_ || !initialized_) return -3;
        return operation();
    }

    template <typename F>
    int64_t release(uint64_t id, F operation) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!id || id != current_) return -1;
        const int64_t result = initialized_ ? operation() : 0;
        // The checked teardown always releases the device, even on failure.
        current_ = 0;
        initialized_ = false;
        cancelled_ = false;
        return result;
    }

    int64_t stop(GgttsAudioControl& control) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!legacyAllowed()) return -1;
        return control.stop() ? 0 : -1;
    }
private:
    std::mutex mutex_;
    uint64_t next_ = 1;
    uint64_t current_ = 0;
    bool cancelled_ = false;
    bool initialized_ = false;
};

#endif
