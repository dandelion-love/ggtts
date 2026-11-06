// Reference side of the codec decoder identity check: calls qwen3-tts.cpp's own AudioTokenizerDecoder (the library,
// not a transcription of it) on a codes file and writes the samples as raw little-endian f32.
//
//   decode_ref <codec.gguf> <codes.json> <out.f32> [threads]
//
// codes.json is the reference CLI's --dump-generated-codes format: {"frames": F, "codebooks": 16, "codes": [...]},
// frame-major. Built by reference/build.sh against a qwen3-tts.cpp checkout at 16bb5af and its CPU build.
#include "audio_tokenizer_decoder.h"
#include "gguf_loader.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static bool read_codes(const char * path, int & frames, int & codebooks, std::vector<int32_t> & codes) {
    std::ifstream in(path);
    if (!in) return false;
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    auto field = [&](const char * key) -> int {
        const size_t k = s.find(key);
        if (k == std::string::npos) return -1;
        return std::atoi(s.c_str() + s.find(':', k) + 1);
    };
    frames = field("\"frames\"");
    codebooks = field("\"codebooks\"");
    const size_t open = s.find('[', s.find("\"codes\""));
    const size_t close = s.find(']', open);
    if (frames <= 0 || codebooks <= 0 || open == std::string::npos || close == std::string::npos) return false;
    codes.clear();
    const char * p = s.c_str() + open + 1;
    const char * end = s.c_str() + close;
    while (p < end) {
        char * next = nullptr;
        const long v = std::strtol(p, &next, 10);
        if (next == p) { ++p; continue; }
        codes.push_back((int32_t) v);
        p = next;
    }
    return (int) codes.size() == frames * codebooks;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <codec.gguf> <codes.json> <out.f32> [threads]\n", argv[0]);
        return 2;
    }
    if (argc > 4) qwen3_tts::set_cpu_thread_count(std::atoi(argv[4]));
    int frames = 0, codebooks = 0;
    std::vector<int32_t> codes;
    if (!read_codes(argv[2], frames, codebooks, codes)) {
        std::fprintf(stderr, "cannot read codes from %s\n", argv[2]);
        return 1;
    }
    qwen3_tts::AudioTokenizerDecoder dec;
    if (!dec.load_model(argv[1])) {
        std::fprintf(stderr, "load: %s\n", dec.get_error().c_str());
        return 1;
    }
    std::vector<float> samples;
    if (!dec.decode(codes.data(), frames, samples)) {
        std::fprintf(stderr, "decode: %s\n", dec.get_error().c_str());
        return 1;
    }
    FILE * f = std::fopen(argv[3], "wb");
    if (!f || std::fwrite(samples.data(), sizeof(float), samples.size(), f) != samples.size()) return 1;
    std::fclose(f);
    std::fprintf(stderr, "decode_ref: %d frames -> %zu samples, compute %lld ms\n", frames, samples.size(),
                 (long long) dec.get_last_timing().graph_compute_ms);
    return 0;
}
