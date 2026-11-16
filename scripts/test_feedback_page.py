import base64
import copy
import unittest

from scripts.feedback_page import generate, validate_input, validate_result

def source():
    return {"format":"ggtts-feedback-input/1","id":"test-1","items":[
        {"id":"clip-1","text":"A中😀𠀀，文","audio":"data:audio/flac;base64,"+base64.b64encode(b"fLaC\x80\x00\x00\x22" + bytes(10) + ((24000 << 44) | 1).to_bytes(8,"big") + bytes(16)).decode()}
    ]}

def result():
    return {"format":"ggtts-feedback-result/1","id":"test-1","note":"","items":[
        {"id":"clip-1","marks":[0,1,2],"rating":"unsure","note":"note"}
    ]}

class ContractTest(unittest.TestCase):
    def test_round_trip_and_supplementary_cjk(self):
        validate_input(source())
        self.assertEqual(validate_result(result(),source()),result())

    def test_planted_wrong_index_is_refused(self):
        bad=result()
        bad["items"][0]["marks"]=[3]
        with self.assertRaisesRegex(ValueError,"index"):
            validate_result(bad,source())
        bad["items"][0]["marks"]=[True]
        with self.assertRaisesRegex(ValueError,"index"):
            validate_result(bad,source())

    def test_missing_duplicate_and_wrong_identity_refused(self):
        for mutation in ("missing", "duplicate", "wrong-id", "rating"):
            bad=result()
            if mutation=="missing": bad["items"]=[]
            if mutation=="duplicate": bad["items"].append(copy.deepcopy(bad["items"][0]))
            if mutation=="wrong-id": bad["items"][0]["id"]="other"
            if mutation=="rating": bad["items"][0]["rating"]="unknown"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_result(bad,source())

    def test_audio_external_invalid_and_non_flac_refused(self):
        for audio in ("https://example.invalid/a.flac","data:audio/flac;base64,!!!","data:audio/flac;base64,QUJD","data:audio/flac;base64,ZkxhQw=="):
            bad=source(); bad["items"][0]["audio"]=audio
            with self.subTest(audio=audio), self.assertRaises(ValueError):
                validate_input(bad)

    def test_markup_cannot_break_out_of_embedded_json(self):
        doc=source(); doc["items"][0]["text"]="</script><script>alert(1)</script>中"
        html=generate(doc, "<html><!-- GGTTS_INPUT --></html>")
        self.assertNotIn("</script><script>", html)
        self.assertIn("\\u003c/script", html)
        with self.assertRaises(ValueError):
            generate(doc, "<html></html>")

if __name__=="__main__":
    unittest.main()
