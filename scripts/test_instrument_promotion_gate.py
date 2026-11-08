import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import instrument_promotion_gate as gate


class InstrumentGateTest(unittest.TestCase):
    def test_missing_mismatch_and_skipped_test_never_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            scratch, libs = root / "scratch", root / "libs"
            scratch.mkdir()
            libs.mkdir()
            talker, codec = root / "talker", root / "codec"
            talker.write_bytes(b"talker")
            codec.write_bytes(b"codec")
            sha = lambda b: hashlib.sha256(b).hexdigest()
            def command(cmd, _cwd, log, _env):
                case = next((x for x in cmd if x.startswith(("each_event_", "stop_trace_", "a_hook_boundary_"))), None)
                log.write_text(f"test {case} ... ok\ntest result: ok. 1 passed; 0 failed\n" if case else "build ok\n")
                return 0
            with patch.object(gate, "TALKER_SHA", sha(b"talker")), patch.object(gate, "CODEC_SHA", sha(b"codec")):
                ok = gate.run(root, scratch, talker, codec, libs, True, 2, command)
                self.assertEqual([ok[x]["status"] for x in ("cpu", "cuda")], ["pass", "pass"])
                unavailable = gate.run(root, scratch, talker, codec, libs, False, 2, command)
                self.assertEqual(unavailable["cuda"]["status"], "incomplete")
                missing = gate.run(root, scratch, None, codec, libs, True, 2, command)
                self.assertEqual(missing["cpu"]["status"], "incomplete")
                codec.write_bytes(b"wrong")
                self.assertEqual(gate.run(root, scratch, talker, codec, libs, True, 2, command)["cpu"]["status"], "fail")
                codec.write_bytes(b"codec")
                def no_test(cmd, _cwd, log, _env):
                    log.write_text("running 0 tests\n")
                    return 0
                absent = gate.run(root, scratch, talker, codec, libs, True, 2, no_test)
                self.assertEqual(absent["cpu"]["status"], "fail")
                self.assertEqual(absent["cuda"]["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
