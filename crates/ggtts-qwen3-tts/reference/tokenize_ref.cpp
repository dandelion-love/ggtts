// Reference side of the text tokenizer identity check: calls qwen3-tts.cpp's own TextTokenizer (the library, not a
// transcription of it) on each input record and prints the token ids.
//
//   tokenize_ref <talker.gguf> <records.txt>
//
// One record per line. Inside a record `\n`, `\r`, `\t` and `\\` stand for a newline, carriage return, tab and
// backslash, so a record can carry the whitespace the pre-tokenizer splits on; every other byte is passed through
// unchanged, invalid UTF-8 included. For record i the output line is
//
//   i <TAB> encode <TAB> encode_for_tts <TAB> encode_instruct <TAB> encode_reference_for_tts
//
// each field the ids joined by single spaces. examples/tokenize.rs prints the same format. Built against a
// qwen3-tts.cpp checkout at 16bb5af and its CPU build:
//
//   g++ -O2 -std=c++17 -I$QWEN3_TTS_CPP/src -I$QWEN3_TTS_CPP/ggml/include reference/tokenize_ref.cpp \
//     $QWEN3_TTS_BUILD/libtext_tokenizer.a $QWEN3_TTS_BUILD/ggml/src/libggml-base.a -fopenmp -lpthread -o tokenize_ref
#include "text_tokenizer.h"

#include "gguf.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static std::string unescape(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[i + 1];
            if (c == 'n' || c == 'r' || c == 't' || c == '\\') {
                out += c == 'n' ? '\n' : c == 'r' ? '\r' : c == 't' ? '\t' : '\\';
                ++i;
                continue;
            }
        }
        out += s[i];
    }
    return out;
}

static void field(const std::vector<int32_t> & ids) {
    std::putchar('\t');
    for (size_t i = 0; i < ids.size(); ++i) {
        std::printf(i ? " %d" : "%d", ids[i]);
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <talker.gguf> <records.txt>\n", argv[0]);
        return 2;
    }
    gguf_init_params params = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(argv[1], params);
    if (!ctx) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    qwen3_tts::TextTokenizer tok;
    if (!tok.load_from_gguf(ctx)) {
        std::fprintf(stderr, "load: %s\n", tok.get_error().c_str());
        return 1;
    }
    std::ifstream in(argv[2], std::ios::binary);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[2]);
        return 1;
    }
    std::string line;
    int i = 0;
    while (std::getline(in, line)) {
        const std::string text = unescape(line);
        std::printf("%d", i++);
        field(tok.encode(text));
        field(tok.encode_for_tts(text));
        field(tok.encode_instruct(text));
        field(tok.encode_reference_for_tts(text));
        std::putchar('\n');
    }
    gguf_free(ctx);
    return 0;
}
