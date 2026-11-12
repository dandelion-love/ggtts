"""The Python binding, on the fake engine (CI) and on real models (opt-in).

Real models, as the C-ABI's tests take them:

    GGTTS_TALKER=<custom_voice 1.7B talker with two speakers> GGTTS_CODEC=<codec> \
    GGTTS_BASE_TALKER=<a Base talker, for the instruction refusal> pytest crates/ggtts-py/tests
"""

import array
import os
import threading
import time

import pytest

import ggtts

TEXT = "今日天氣好好，我哋去海邊行下。\n行完之後，去食飯。"
LINE = "今日天氣好好，我哋去海邊行下。"
INSTRUCTION = "Speak slowly and softly."

planted = pytest.mark.skipif("panic" not in ggtts.ENGINES, reason="the planted engines need a build with --features plant")


def fake():
    return ggtts.load("fake")


def code_of(excinfo):
    return excinfo.value.code


# -- loading and rendering ---------------------------------------------------------------------------------------


def test_a_fake_load_and_render():
    with fake() as tts:
        assert (tts.engine, tts.device, tts.speaker, tts.instruction) == ("fake", "cpu", None, None)
        assert tts.info["family"] == "fake"
        assert tts.model_files == []
        audio = tts.render(TEXT, seed=7)
        assert audio.sample_rate > 0 and len(audio) > 0
        assert isinstance(audio.pcm, bytearray) and len(audio.pcm) == 4 * len(audio)
        # The same text and seed give the same samples on a fresh engine (the fake engine counts its draws per text).
        with fake() as again:
            assert again.render(TEXT, seed=7).pcm == audio.pcm
        assert audio.duration == pytest.approx(len(audio) / audio.sample_rate)
    assert tts.closed


def test_samples_are_float32_numpy_or_array():
    np = pytest.importorskip("numpy")
    with fake() as tts:
        audio = tts.render(TEXT)
    samples = audio.samples
    assert isinstance(samples, np.ndarray) and samples.dtype == np.float32 and samples.shape == (len(audio),)
    assert np.all(np.isfinite(samples)) and np.abs(samples).max() > 0
    assert samples.tobytes() == audio.to_bytes()


def test_samples_without_numpy_are_an_array_of_float32(monkeypatch):
    with fake() as tts:
        audio = tts.render(TEXT)
    monkeypatch.setattr(ggtts, "_np", None)
    samples = audio.samples
    assert isinstance(samples, array.array) and samples.typecode == "f" and len(samples) == len(audio)
    assert samples.tobytes() == audio.to_bytes()
    with pytest.raises(ImportError):
        audio.numpy()


def test_write_flac(tmp_path):
    with fake() as tts:
        audio = tts.render(TEXT)
    written = audio.write_flac(tmp_path / "out.wav")
    assert written.endswith("out.flac")
    with open(written, "rb") as f:
        assert f.read(4) == b"fLaC"
    # Any float32 source, with a rate.
    other = ggtts.write_flac(tmp_path / "list", [0.0, 0.25, -0.25], 24000)
    assert open(other, "rb").read(4) == b"fLaC"
    with pytest.raises(ggtts.GgttsError) as e:
        ggtts.write_flac(tmp_path / "bad", b"\x00\x00\x00", 24000)
    assert code_of(e) == ggtts.ERR_ARG
    with pytest.raises(TypeError):
        ggtts.write_flac(tmp_path / "norate", [0.0])


# -- streaming ---------------------------------------------------------------------------------------------------


def test_a_stream_yields_the_chunks_in_order():
    with fake() as tts:
        whole = tts.render(TEXT)
        chunks = list(tts.stream(TEXT))
        assert len(chunks) >= 2
        assert sum(len(c) for c in chunks) == len(whole)
        assert all(c.sample_rate == whole.sample_rate for c in chunks)
        # The callback form gives the same chunks.
        seen = []
        assert tts.render_stream(TEXT, seen.append) is True
        assert [c.pcm for c in seen] == [c.pcm for c in chunks]


def test_a_stream_can_be_stopped():
    with fake() as tts:
        seen = []
        assert tts.render_stream(TEXT, lambda c: seen.append(c) or False) is False
        assert len(seen) == 1
        for chunk in tts.stream(TEXT):
            break
        # The engine is free again once the iterator is stopped.
        assert len(tts.render(TEXT)) > 0


def test_an_exception_in_the_callback_stops_and_is_raised():
    class Mine(Exception):
        pass

    def boom(_):
        raise Mine("from the callback")

    with fake() as tts:
        with pytest.raises(Mine):
            tts.render_stream(TEXT, boom)
        assert len(tts.render(TEXT)) > 0


def test_a_call_from_the_engines_own_callback_is_busy():
    with fake() as tts:
        inner = []

        def again(_):
            with pytest.raises(ggtts.GgttsError) as e:
                tts.render(TEXT)
            inner.append(code_of(e))
            return False

        tts.render_stream(TEXT, again)
        assert inner == [ggtts.ERR_BUSY]


# -- refusals ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["？！", "好。\n？！", "  \n ", ""])
def test_text_with_nothing_to_say_is_refused_before_rendering(text):
    with fake() as tts:
        with pytest.raises(ggtts.GgttsError) as e:
            tts.render(text)
        assert code_of(e) == ggtts.ERR_ARG
        seen = []
        with pytest.raises(ggtts.GgttsError) as e:
            tts.render_stream(text, seen.append)
        assert code_of(e) == ggtts.ERR_ARG and seen == []
        with pytest.raises(ggtts.GgttsError) as e:
            list(tts.stream(text))
        assert code_of(e) == ggtts.ERR_ARG


@pytest.mark.parametrize(
    "kwargs, code, fragment",
    [
        (dict(engine="fake", speaker="hk_voice_343"), ggtts.ERR_ARG, "only supported by the qwen3-tts engine"),
        (dict(engine="fake", instruction=INSTRUCTION), ggtts.ERR_ARG, "takes no style instruction"),
        (dict(engine="kitten-tts-2", instruction=INSTRUCTION), ggtts.ERR_ARG, "takes no style instruction"),
        (dict(engine="fake", device="tpu"), ggtts.ERR_ARG, "device must be"),
        (dict(engine="fake", device="cuda"), ggtts.ERR_ARG, "CPU only"),
        (dict(engine="nonesuch"), ggtts.ERR_ENGINE, "no engine named"),
        (dict(engine="kitten-tts-2"), ggtts.ERR_ARG, "needs a model"),
    ],
)
def test_an_invalid_argument_or_refusal_is_a_code_and_a_message(kwargs, code, fragment):
    with pytest.raises(ggtts.GgttsError) as e:
        ggtts.load(**kwargs)
    assert code_of(e) == code
    assert fragment in e.value.message
    assert fragment in str(e.value) and f"(code {code})" in str(e.value)


def test_empty_settings_are_no_settings():
    with ggtts.load("fake", speaker="", instruction="") as tts:
        assert (tts.speaker, tts.instruction) == (None, None)


def test_a_file_that_is_not_a_model_is_refused(tmp_path):
    bad = tmp_path / "not-a-model.gguf"
    bad.write_bytes(b"not a model")
    with pytest.raises(ggtts.GgttsError) as e:
        ggtts.load("qwen3-tts", model=bad, codec=bad)
    assert code_of(e) == ggtts.ERR_ENGINE and "not a GGUF file" in e.value.message


def test_offline_with_an_empty_cache_downloads_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("GGTTS_CACHE_DIR", str(tmp_path / "cache"))
    with pytest.raises(ggtts.GgttsError) as e:
        ggtts.load(offline=True)
    assert code_of(e) == ggtts.ERR_ENGINE
    if "qwen3-tts" in ggtts.ENGINES:
        assert "offline mode does not download" in e.value.message
    else:
        assert "built without the ggml feature" in e.value.message
    assert not any((tmp_path / "cache").rglob("*.gguf"))


def test_a_closed_engine_is_a_handle_error():
    tts = fake()
    tts.close()
    with pytest.raises(ggtts.GgttsError) as e:
        tts.render(TEXT)
    assert code_of(e) == ggtts.ERR_HANDLE


def test_the_codes_are_the_c_abi_codes():
    assert (ggtts.OK, ggtts.ERR_ARG, ggtts.ERR_HANDLE, ggtts.ERR_ENGINE, ggtts.ERR_PANIC, ggtts.ERR_STOPPED,
            ggtts.ERR_BUSY) == (0, -1, -2, -3, -4, -5, -6)
    assert ggtts.DEFAULT_SEED == 20260905
    assert issubclass(ggtts.GgttsError, Exception)


# -- the boundary: panics and the GIL ----------------------------------------------------------------------------


@planted
def test_a_panic_never_crosses_into_python():
    tts = ggtts.load("panic")
    with pytest.raises(ggtts.GgttsError) as e:
        tts.render(TEXT)
    assert code_of(e) == ggtts.ERR_PANIC and "planted panic in generate" in e.value.message
    # The engine that panicked is dropped.
    assert tts.closed
    with pytest.raises(ggtts.GgttsError) as e:
        tts.render(TEXT)
    assert code_of(e) == ggtts.ERR_HANDLE
    with pytest.raises(ggtts.GgttsError) as e:
        ggtts.load("panic-load")
    assert code_of(e) == ggtts.ERR_PANIC and "planted panic in load" in e.value.message


@planted
def test_the_gil_is_released_while_rendering():
    """The `slow` engine takes a quarter second per chunk. While another thread renders, this thread keeps running:
    with the GIL held for the render it would not run at all until the render returned."""
    tts = ggtts.load("slow")
    started = threading.Event()
    elapsed = []

    def render():
        started.set()
        t = time.perf_counter()
        tts.render(TEXT)
        elapsed.append(time.perf_counter() - t)

    worker = threading.Thread(target=render)
    worker.start()
    started.wait()
    ticks = 0
    while worker.is_alive():
        ticks += 1
    worker.join()
    tts.close()
    assert elapsed and elapsed[0] >= 0.4, elapsed
    assert ticks > 1000, f"{ticks} iterations of this thread during a {elapsed[0]:.2f} s render"


@planted
def test_a_render_on_another_thread_meanwhile_is_busy():
    with ggtts.load("slow") as tts:
        worker = threading.Thread(target=tts.render, args=(TEXT,))
        worker.start()
        time.sleep(0.1)
        with pytest.raises(ggtts.GgttsError) as e:
            tts.render(TEXT)
        assert code_of(e) == ggtts.ERR_BUSY
        worker.join()


# -- real models, opt-in -----------------------------------------------------------------------------------------


def env(name):
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to run the real-model tests")
    if "qwen3-tts" not in ggtts.ENGINES:
        pytest.skip("this build has no model engines (build with --features ggml)")
    return value


def qwen(**kwargs):
    return ggtts.load("qwen3-tts", model=env("GGTTS_TALKER"), codec=env("GGTTS_CODEC"), offline=True, **kwargs)


def speakers():
    with pytest.raises(ggtts.GgttsError) as e:
        qwen(speaker="no_such_speaker")
    msg = e.value.message
    assert code_of(e) == ggtts.ERR_ENGINE and "no speaker no_such_speaker" in msg, msg
    assert "--speaker" not in msg, msg
    return msg.split("speakers are: ")[1].split(". ")[0].split(", ")


def test_real_speakers_reach_the_engine():
    names = speakers()
    assert len(names) >= 2, names
    with qwen(speaker=names[0]) as a, qwen(speaker=names[1]) as b:
        x, y = a.render(LINE), b.render(LINE)
    assert len(x) > 0 and x.pcm != y.pcm


def test_real_instruction_reaches_the_engine_and_a_stream_matches_the_render():
    speaker = speakers()[0]
    with qwen(speaker=speaker) as plain, qwen(speaker=speaker, instruction=INSTRUCTION) as styled:
        p = plain.render(LINE)
        assert styled.instruction == INSTRUCTION
        assert styled.render(LINE).pcm != p.pcm
        chunks = list(plain.stream(LINE))
        assert sum(len(c) for c in chunks) == len(p)


def test_real_default_model_from_the_cache(monkeypatch):
    """GGTTS_DEFAULT_CACHE: a cache directory holding the default model, as ggtts-render downloads it."""
    cache = env("GGTTS_DEFAULT_CACHE")
    monkeypatch.setenv("GGTTS_CACHE_DIR", cache)
    with ggtts.load(offline=True) as tts:
        assert tts.speaker == "hk_voice_343"
        assert [(f["role"], f["source"]) for f in tts.model_files] == [("talker", "cache"), ("codec", "cache")]
        assert len(tts.render(LINE)) > 0


def test_real_base_checkpoint_refuses_an_instruction():
    base = env("GGTTS_BASE_TALKER")
    with pytest.raises(ggtts.GgttsError) as e:
        ggtts.load("qwen3-tts", model=base, codec=env("GGTTS_CODEC"), instruction=INSTRUCTION)
    assert code_of(e) == ggtts.ERR_ENGINE
    assert "base model" in e.value.message and "takes no style instruction" in e.value.message
