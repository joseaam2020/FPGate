#define _GNU_SOURCE
#include "image_prep.h"

#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <jerror.h>
#include <jpeglib.h>
#include <png.h>
#include <webp/decode.h>

#include "resample.h"

#define JPEG_MAX_MEMORY (96L * 1024 * 1024)

const char *prep_status_name(prep_status_t s)
{
    switch (s) {
    case PREP_OK:          return "ok";
    case PREP_CORRUPT:     return "corrupt";
    case PREP_UNSUPPORTED: return "unsupported";
    case PREP_TOO_LARGE:   return "too_large";
    default:               return "nomem";
    }
}

void prep_result_free(prep_result_t *r)
{
    if (r->rgb)
        explicit_bzero(r->rgb, r->rgb_len);
    free(r->rgb);
    memset(r, 0, sizeof *r);
}

/* ---- JPEG ------------------------------------------------------------- */

struct jpeg_ctx {
    struct jpeg_error_mgr pub;
    jmp_buf jb;
    int damaged;                /* a warning that means the data is cut or garbled */
};

static void jpeg_fatal(j_common_ptr c)
{
    longjmp(((struct jpeg_ctx *)c->err)->jb, 1);
}

static void jpeg_warn(j_common_ptr c, int level)
{
    if (level >= 0)
        return;
    switch (c->err->msg_code) {
    case JWRN_JPEG_EOF:         /* premature end: libjpeg would pad with grey */
    case JWRN_HIT_MARKER:
    case JWRN_HUFF_BAD_CODE:
        ((struct jpeg_ctx *)c->err)->damaged = 1;
        break;
    default:
        break;
    }
}

static prep_status_t decode_jpeg(const uint8_t *data, size_t len, const prep_cfg_t *cfg,
                                 prep_result_t *out)
{
    struct jpeg_decompress_struct cinfo;
    struct jpeg_ctx ctx;
    resizer_t *volatile rs = NULL;
    uint8_t *volatile row = NULL;
    uint8_t *volatile rgb = NULL;
    volatile size_t row_len = 0;
    volatile prep_status_t st = PREP_CORRUPT;

    memset(&cinfo, 0, sizeof cinfo);
    cinfo.err = jpeg_std_error(&ctx.pub);
    ctx.pub.error_exit = jpeg_fatal;
    ctx.pub.emit_message = jpeg_warn;
    ctx.damaged = 0;

    if (setjmp(ctx.jb)) {
        jpeg_destroy_decompress(&cinfo);
        goto done;
    }
    jpeg_create_decompress(&cinfo);
    cinfo.mem->max_memory_to_use = JPEG_MAX_MEMORY;
    jpeg_mem_src(&cinfo, data, (unsigned long)len);
    jpeg_read_header(&cinfo, TRUE);

    if (cinfo.jpeg_color_space == JCS_CMYK || cinfo.jpeg_color_space == JCS_YCCK ||
        cinfo.data_precision != 8) {
        jpeg_destroy_decompress(&cinfo);
        st = PREP_UNSUPPORTED;
        goto done;
    }
    if ((unsigned long)cinfo.image_width * cinfo.image_height > cfg->max_pixels) {
        jpeg_destroy_decompress(&cinfo);
        st = PREP_TOO_LARGE;
        goto done;
    }
    out->src_w = (int)cinfo.image_width;
    out->src_h = (int)cinfo.image_height;

    cinfo.out_color_space = JCS_RGB;
    cinfo.dct_method = JDCT_ISLOW;          /* the default, and what PIL uses */
    cinfo.do_fancy_upsampling = TRUE;
    if (cfg->dct_scaling) {
        for (int denom = 8; denom > 1; denom /= 2) {
            cinfo.scale_num = 1;
            cinfo.scale_denom = denom;
            jpeg_calc_output_dimensions(&cinfo);
            unsigned min_side = cinfo.output_width < cinfo.output_height
                                    ? cinfo.output_width : cinfo.output_height;
            if (min_side >= (unsigned)cfg->size)
                break;
            cinfo.scale_denom = 1;
        }
    }

    jpeg_start_decompress(&cinfo);
    rs = rs_new((int)cinfo.output_width, (int)cinfo.output_height, cfg->size, cfg->center_crop);
    row_len = (size_t)cinfo.output_width * 3;
    row = malloc(row_len);
    if (!rs || !row) {
        jpeg_destroy_decompress(&cinfo);
        st = PREP_NOMEM;
        goto done;
    }
    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW rp = row;
        jpeg_read_scanlines(&cinfo, &rp, 1);
        rs_push_row(rs, row);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);

    if (ctx.damaged) {
        st = PREP_CORRUPT;
        goto done;
    }
    rgb = malloc((size_t)cfg->size * (size_t)cfg->size * 3);
    if (!rgb || rs_finish(rs, rgb) != 0) {
        st = PREP_NOMEM;
        goto done;
    }
    out->rgb = rgb;
    out->rgb_len = (size_t)cfg->size * (size_t)cfg->size * 3;
    rgb = NULL;
    st = PREP_OK;

done:
    if (row) {
        explicit_bzero(row, row_len);
        free(row);
    }
    free(rgb);
    rs_free(rs);
    return st;
}

/* ---- PNG -------------------------------------------------------------- */

struct png_src {
    const uint8_t *p;
    size_t len, pos;
};

static void png_read_cb(png_structp png, png_bytep dst, size_t n)
{
    struct png_src *s = png_get_io_ptr(png);
    if (s->len - s->pos < n)
        png_error(png, "truncated");
    memcpy(dst, s->p + s->pos, n);
    s->pos += n;
}

static void png_fatal(png_structp png, png_const_charp msg)
{
    (void)msg;
    longjmp(png_jmpbuf(png), 1);
}

static void png_warn(png_structp png, png_const_charp msg)
{
    (void)png;
    (void)msg;
}

/* Converts one decoded row (1..4 channels, 8 bit) to RGB, dropping alpha. */
static void row_to_rgb(const uint8_t *src, uint8_t *dst, png_uint_32 w, int ch)
{
    switch (ch) {
    case 1:
        for (png_uint_32 i = 0; i < w; i++)
            dst[i * 3] = dst[i * 3 + 1] = dst[i * 3 + 2] = src[i];
        break;
    case 2:
        for (png_uint_32 i = 0; i < w; i++)
            dst[i * 3] = dst[i * 3 + 1] = dst[i * 3 + 2] = src[i * 2];
        break;
    case 3:
        memcpy(dst, src, (size_t)w * 3);
        break;
    default:
        for (png_uint_32 i = 0; i < w; i++) {
            dst[i * 3] = src[i * 4];
            dst[i * 3 + 1] = src[i * 4 + 1];
            dst[i * 3 + 2] = src[i * 4 + 2];
        }
        break;
    }
}

static prep_status_t decode_png(const uint8_t *data, size_t len, const prep_cfg_t *cfg,
                                prep_result_t *out)
{
    struct png_src src = { data, len, 0 };
    png_structp png = NULL;
    png_infop info = NULL;
    resizer_t *volatile rs = NULL;
    uint8_t *volatile rowbuf = NULL;
    uint8_t *volatile rgbrow = NULL;
    uint8_t *volatile full = NULL;
    uint8_t *volatile rgb = NULL;
    volatile size_t full_len = 0, row_len = 0;
    volatile prep_status_t st = PREP_CORRUPT;

    /* Dimensions sit at a fixed place in the first chunk: refuse bombs before
     * libpng allocates anything. */
    if (len >= 24 && memcmp(data + 12, "IHDR", 4) == 0) {
        unsigned long iw = ((unsigned long)data[16] << 24) | (data[17] << 16) | (data[18] << 8) | data[19];
        unsigned long ih = ((unsigned long)data[20] << 24) | (data[21] << 16) | (data[22] << 8) | data[23];
        if (iw && ih && (iw > 65535 || ih > 65535 || iw * ih > cfg->max_pixels))
            return PREP_TOO_LARGE;
    }

    png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, png_fatal, png_warn);
    if (!png)
        return PREP_NOMEM;
    info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, NULL, NULL);
        return PREP_NOMEM;
    }
    if (setjmp(png_jmpbuf(png)))
        goto done;

    png_set_read_fn(png, &src, png_read_cb);
    png_set_user_limits(png, 65535, 65535);
    png_set_chunk_malloc_max(png, 8 * 1024 * 1024);
    png_read_info(png, info);

    png_uint_32 w = png_get_image_width(png, info), h = png_get_image_height(png, info);
    if ((unsigned long)w * h > cfg->max_pixels) {
        st = PREP_TOO_LARGE;
        goto done;
    }
    out->src_w = (int)w;
    out->src_h = (int)h;

    int ctype = png_get_color_type(png, info);
    if (ctype == PNG_COLOR_TYPE_PALETTE)
        png_set_palette_to_rgb(png);
    if (ctype == PNG_COLOR_TYPE_GRAY && png_get_bit_depth(png, info) < 8)
        png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_bit_depth(png, info) == 16)
        png_set_strip_16(png);
    int passes = png_set_interlace_handling(png);
    png_read_update_info(png, info);

    int ch = png_get_channels(png, info);
    size_t rowbytes = png_get_rowbytes(png, info);
    if (ch < 1 || ch > 4 || png_get_bit_depth(png, info) != 8) {
        st = PREP_UNSUPPORTED;
        goto done;
    }

    rs = rs_new((int)w, (int)h, cfg->size, cfg->center_crop);
    rgbrow = malloc((size_t)w * 3);
    row_len = rowbytes;
    rowbuf = malloc(rowbytes);
    if (!rs || !rgbrow || !rowbuf) {
        st = PREP_NOMEM;
        goto done;
    }

    if (passes > 1) {                       /* Adam7: rows arrive over several passes */
        full_len = rowbytes * h;
        full = malloc(full_len);
        if (!full) {
            st = PREP_NOMEM;
            goto done;
        }
        for (int p = 0; p < passes; p++)
            for (png_uint_32 y = 0; y < h; y++) {
                png_bytep rp = full + (size_t)y * rowbytes;
                png_read_row(png, rp, NULL);
            }
        for (png_uint_32 y = 0; y < h; y++) {
            row_to_rgb(full + (size_t)y * rowbytes, rgbrow, w, ch);
            rs_push_row(rs, rgbrow);
        }
    } else {
        for (png_uint_32 y = 0; y < h; y++) {
            png_read_row(png, rowbuf, NULL);
            row_to_rgb(rowbuf, rgbrow, w, ch);
            rs_push_row(rs, rgbrow);
        }
    }

    rgb = malloc((size_t)cfg->size * (size_t)cfg->size * 3);
    if (!rgb || rs_finish(rs, rgb) != 0) {
        st = PREP_NOMEM;
        goto done;
    }
    out->rgb = rgb;
    out->rgb_len = (size_t)cfg->size * (size_t)cfg->size * 3;
    rgb = NULL;
    st = PREP_OK;

done:
    png_destroy_read_struct(&png, &info, NULL);
    if (rowbuf) {
        explicit_bzero(rowbuf, row_len);
        free(rowbuf);
    }
    if (rgbrow) {
        explicit_bzero(rgbrow, (size_t)out->src_w * 3);
        free(rgbrow);
    }
    if (full) {
        explicit_bzero(full, full_len);
        free(full);
    }
    free(rgb);
    rs_free(rs);
    return st;
}

/* ---- WebP ------------------------------------------------------------- */

static prep_status_t decode_webp(const uint8_t *data, size_t len, const prep_cfg_t *cfg,
                                 prep_result_t *out)
{
    WebPBitstreamFeatures f;
    VP8StatusCode vs = WebPGetFeatures(data, len, &f);
    if (vs == VP8_STATUS_UNSUPPORTED_FEATURE)
        return PREP_UNSUPPORTED;
    if (vs != VP8_STATUS_OK)
        return PREP_CORRUPT;            /* truncated or damaged header */
    if (f.has_animation)
        return PREP_UNSUPPORTED;        /* only still images, like JPEG and PNG */
    if (f.width <= 0 || f.height <= 0 ||
        (unsigned long)f.width * (unsigned long)f.height > cfg->max_pixels)
        return PREP_TOO_LARGE;

    int w = f.width, h = f.height;
    out->src_w = w;
    out->src_h = h;

    /* Alpha is dropped without compositing, as PIL's convert("RGB") does. */
    uint8_t *px = WebPDecodeRGB(data, len, &w, &h);
    if (!px)
        return PREP_CORRUPT;            /* the bitstream does not decode completely */

    prep_status_t st = PREP_NOMEM;
    size_t px_len = (size_t)w * (size_t)h * 3;
    resizer_t *rs = rs_new(w, h, cfg->size, cfg->center_crop);
    uint8_t *rgb = rs ? malloc((size_t)cfg->size * (size_t)cfg->size * 3) : NULL;
    if (rs && rgb) {
        for (int y = 0; y < h; y++)
            rs_push_row(rs, px + (size_t)y * (size_t)w * 3);
        if (rs_finish(rs, rgb) == 0) {
            out->rgb = rgb;
            out->rgb_len = (size_t)cfg->size * (size_t)cfg->size * 3;
            rgb = NULL;
            st = PREP_OK;
        }
    }
    free(rgb);
    rs_free(rs);
    explicit_bzero(px, px_len);
    WebPFree(px);
    return st;
}

prep_status_t image_prep(img_kind_t kind, const uint8_t *data, size_t len,
                         const prep_cfg_t *cfg, prep_result_t *out)
{
    memset(out, 0, sizeof *out);
    switch (kind) {
    case IMG_JPEG: return decode_jpeg(data, len, cfg, out);
    case IMG_PNG:  return decode_png(data, len, cfg, out);
    case IMG_WEBP: return decode_webp(data, len, cfg, out);
    default:       return PREP_UNSUPPORTED;
    }
}
