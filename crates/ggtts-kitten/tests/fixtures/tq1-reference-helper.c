/* Compile and link against the pinned ggml source for independent fixture regeneration. */
#include <stdint.h>
#include <stdio.h>
extern void quantize_row_tq1_0_ref(const float *, void *, int64_t);
extern void dequantize_row_tq1_0(const void *, float *, int64_t);
int main(void) {
    float x[256], y[256];
    unsigned char b[54];
    while (fread(x, sizeof(float), 256, stdin) == 256) {
        quantize_row_tq1_0_ref(x, b, 256);
        dequantize_row_tq1_0(b, y, 256);
        if (fwrite(b, 1, 54, stdout) != 54 || fwrite(y, sizeof(float), 256, stdout) != 256) return 1;
    }
    return ferror(stdin) ? 1 : 0;
}
