/* Compare the committed reference outputs with a separately built ggml. */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern void dequantize_row_tq1_0(const void *, float *, int64_t);

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 2;
    unsigned char block[54], bad[54];
    float expected[256], actual[256];
    int count = 0, bad_code = 0, bad_scale = 0;
    while (fread(block, 1, sizeof block, f) == sizeof block) {
        if (fread(expected, 1, sizeof expected, f) != sizeof expected) return 2;
        dequantize_row_tq1_0(block, actual, 256);
        if (memcmp(expected, actual, sizeof actual)) return 1;
        memcpy(bad, block, sizeof bad); bad[0] ^= 0xff;
        dequantize_row_tq1_0(bad, actual, 256);
        bad_code += memcmp(expected, actual, sizeof actual) != 0;
        memcpy(bad, block, sizeof bad); bad[53] ^= 1;
        dequantize_row_tq1_0(bad, actual, 256);
        bad_scale += memcmp(expected, actual, sizeof actual) != 0;
        ++count;
    }
    if (ferror(f) || !feof(f) || ftell(f) != 64L * (54 + 1024)) return 2;
    fclose(f);
    printf("blocks=%d code-controls=%d scale-controls=%d\n", count, bad_code, bad_scale);
    return count == 64 && bad_code && bad_scale ? 0 : 1;
}
