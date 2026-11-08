/* Image detection from magic numbers, the Content-Type classifier and the Accept rewrite. */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "image_detect.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    (void)image_sniff(data, size);
    (void)image_classify_content_type((const char *)data, size);

    char out[4096 + 8];
    size_t n = size > 4096 ? 4096 : size, outn = 0;
    int r = accept_drop_avif((const char *)data, n, out, sizeof out, &outn);
    if (r == 1 && outn > n)
        abort();                    /* removing an entry can never make the value longer */
    return 0;
}
