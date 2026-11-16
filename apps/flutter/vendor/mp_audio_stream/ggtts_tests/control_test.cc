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
    Fake f;
    GgttsAudioControl c(f);
    assert(!c.pause());
    assert(c.begin());
    assert(!c.begin());
    assert(c.push([] { return 7; }) == 7);
    assert(!c.softwareEmpty());
    assert(c.pause());
    assert(c.push([] { assert(false); return 0; }) == -2);
    assert(!c.softwareEmpty());
    assert(c.resume());
    assert(c.stop());
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

}
