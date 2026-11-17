#include "../src/ggtts_audio_control.h"
#include <cassert>
#include <atomic>
#include <thread>

struct Fake : GgttsAudioControl::Backend {
    int starts = 0, stops = 0, clears = 0, destroys = 0;
    uint64_t frames = 8;
    bool fail = false;
    bool start() override { ++starts; return !fail; }
    bool stop() override { ++stops; return !fail; }
    void clear() override { ++clears; frames = 0; }
    void destroy() override { ++destroys; }
    uint64_t queued() override { return frames; }
};

int main() {
    // Plant failure after both resources were created; the production guard
    // must release both, while successful initialization keeps them.
    bool device = false, ring = false;
    auto initialize = [&](bool failAfterRing) {
        GgttsInitGuard guard([&] { device = false; ring = false; });
        device = true;
        ring = true;
        if (failAfterRing) return false;
        guard.commit();
        return true;
    };
    assert(!initialize(true));
    assert(!device && !ring);
    assert(initialize(false));
    assert(device && ring);

    Fake f;
    GgttsAudioControl c(f);
    assert(!c.pause());
    assert(c.queuedFrames() == -1);
    assert(c.begin());
    assert(c.queuedFrames() == 8);
    assert(!c.begin());
    assert(c.push([] { return 7; }) == 7);
    assert(!c.softwareEmpty());
    assert(c.pause());
    assert(c.push([] { assert(false); return 0; }) == -2);
    assert(!c.softwareEmpty());
    assert(c.queuedFrames() == 8);
    assert(c.resume());
    assert(c.stop());
    assert(c.queuedFrames() == -1);
    assert(c.push([] { assert(false); return 0; }) == -3);
    assert(f.clears == 1 && f.frames == 0);
    assert(c.stop() && f.clears == 1);
    assert(c.begin());
    f.fail = true;
    assert(!c.pause());
    assert(c.state() == GgttsAudioControl::State::running);
    f.fail = false;
    assert(c.stop());
    assert(c.teardown() && f.destroys == 1);
    assert(c.begin());
    f.fail = true;
    assert(!c.teardown());
    assert(f.destroys == 2);
    assert(c.state() == GgttsAudioControl::State::idle);
    assert(c.push([] { assert(false); return 0; }) == -3);
    f.fail = false;

    // A stop cannot clear the ring until an in-flight producer leaves.
    assert(c.begin());
    std::atomic<bool> entered{false}, release{false}, finished{false};
    std::thread producer([&] {
        c.push([&] { entered = true; while (!release.load()) std::this_thread::yield(); return 0; });
    });
    while (!entered.load()) std::this_thread::yield();
    std::thread stopper([&] { assert(c.stop()); finished = true; });
    assert(!finished.load());
    release = true;
    producer.join();
    stopper.join();
    assert(finished && c.state() == GgttsAudioControl::State::stopped);
    // Stop admitted during a new initialization must wait for begin and stop
    // that new device, rather than succeeding against the previous stopped state.
    GgttsAudioLifecycle lifecycle;
    std::atomic<bool> initializing{false}, allowBegin{false}, stopAttempt{false}, stopDone{false};
    std::thread initializer([&] {
        std::lock_guard<std::mutex> lock(lifecycle.mutex());
        initializing = true;
        while (!allowBegin.load()) std::this_thread::yield();
        assert(c.begin());
    });
    while (!initializing.load()) std::this_thread::yield();
    std::thread duringInit([&] {
        stopAttempt = true;
        assert(lifecycle.stop(c) == 0);
        stopDone = true;
    });
    while (!stopAttempt.load()) std::this_thread::yield();
    assert(!stopDone.load());
    allowBegin = true;
    initializer.join();
    duringInit.join();
    assert(stopDone && c.state() == GgttsAudioControl::State::stopped);

    // Session stop before initialization rejects admission without opening a
    // device, and a stale stop cannot affect the next reservation.
    Fake sessionBackend;
    GgttsAudioControl sessionControl(sessionBackend);
    GgttsAudioLifecycle sessions;
    const auto first = sessions.reserve();
    assert(first != 0);
    assert(sessions.reserve() == 0);
    assert(sessions.stopSession(first, sessionControl) == 0);
    bool opened = false;
    assert(sessions.initialize(first, [&] { opened = true; return 0; }) != 0);
    assert(!opened);
    assert(sessions.release(first, [&] { assert(false); return 0; }) == 0);
    const auto second = sessions.reserve();
    assert(second != 0 && second != first);
    assert(sessions.stopSession(first, sessionControl) != 0);
    assert(sessions.initialize(second, [&] { return sessionControl.begin() ? 0 : -1; }) == 0);
    assert(sessions.stopSession(first, sessionControl) != 0);
    assert(sessionControl.state() == GgttsAudioControl::State::running);
    assert(sessions.invokeSession(second, [&] { return sessionControl.pause() ? 0 : -1; }) == 0);
    assert(sessions.sessionState(second, sessionControl) == 2);
    assert(sessions.pushSession(second, [&] { return sessionControl.push([] { return 0; }); }) == -2);
    assert(sessions.invokeSession(second, [&] { return sessionControl.resume() ? 0 : -1; }) == 0);
    assert(sessions.stopSession(second, sessionControl) == 0);
    assert(sessions.pushSession(second, [&] { assert(false); return 0; }) == -3);
    assert(sessions.release(second, [&] { return sessionControl.teardown() ? 0 : -1; }) == 0);

    // Stop arriving while the session's initializer holds admission waits and
    // then stops the just-opened device.
    const auto third = sessions.reserve();
    std::atomic<bool> sessionInitializing{false}, sessionAllow{false}, sessionStopAttempt{false};
    std::thread sessionInit([&] {
        assert(sessions.initialize(third, [&] {
            sessionInitializing = true;
            while (!sessionAllow.load()) std::this_thread::yield();
            return sessionControl.begin() ? 0 : -1;
        }) == 0);
    });
    while (!sessionInitializing.load()) std::this_thread::yield();
    std::thread sessionStop([&] {
        sessionStopAttempt = true;
        assert(sessions.stopSession(third, sessionControl) == 0);
    });
    while (!sessionStopAttempt.load()) std::this_thread::yield();
    sessionAllow = true;
    sessionInit.join();
    sessionStop.join();
    assert(sessionControl.state() == GgttsAudioControl::State::stopped);
    assert(sessions.release(third, [&] { return sessionControl.teardown() ? 0 : -1; }) == 0);
    const auto fourth = sessions.reserve();
    assert(sessions.initialize(fourth, [&] { return sessionControl.begin() ? 0 : -1; }) == 0);
    sessionBackend.fail = true;
    assert(sessions.release(fourth, [&] { return sessionControl.teardown() ? 0 : -1; }) == -1);
    assert(sessionControl.state() == GgttsAudioControl::State::idle);
    assert(sessions.sessionState(fourth, sessionControl) == -1);
    assert(sessions.reserve() != 0);

}
