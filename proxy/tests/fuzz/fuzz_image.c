/* The decoders (libjpeg-turbo, libpng, libwebp) and our resampling, fed damaged files. */
#include <stddef.h>
#include <stdint.h>

#include "image_detect.h"
#include "image_prep.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    img_kind_t kind = image_sniff(data, size);
    if (kind != IMG_JPEG && kind != IMG_PNG && kind != IMG_WEBP)
        return 0;
    /* A small target and a pixel cap keep each run short; the format code is the same. */
    prep_cfg_t cfg = {.size = 64, .center_crop = (data[size > 12 ? 12 : 0] & 1) != 0,
                      .dct_scaling = (data[size > 13 ? 13 : 0] & 1) != 0, .max_pixels = 4000000ul};
    prep_result_t r;
    if (image_prep(kind, data, size, &cfg, &r) == PREP_OK)
        prep_result_free(&r);
    return 0;
}
