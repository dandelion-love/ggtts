#ifndef GGTTS_AUDIO_CONTROL_H
#define GGTTS_AUDIO_CONTROL_H

#include <cstdint>
#include <mutex>

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

// Initialization and device transitions take this same admission lock.
class GgttsAudioLifecycle {
public:
    std::mutex& mutex() { return mutex_; }
    int64_t stop(GgttsAudioControl& control) {
        std::lock_guard<std::mutex> lock(mutex_);
        return control.stop() ? 0 : -1;
    }
private:
    std::mutex mutex_;
};

#endif
