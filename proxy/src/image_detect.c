#define _GNU_SOURCE
#include "image_detect.h"

#include <string.h>
#include <strings.h>

ct_class_t image_classify_content_type(const char *value, size_t len)
{
    /* Strip leading spaces and parameters ("image/png; charset=...") */
    while (len && (*value == ' ' || *value == '\t')) {
        value++;
        len--;
    }
    size_t n = 0;
    while (n < len && value[n] != ';' && value[n] != ' ' && value[n] != '\t')
        n++;
    if (n == 0)
        return CT_AMBIGUOUS;

    if (n >= 6 && strncasecmp(value, "image/", 6) == 0) {
        /* SVG is XML, not a raster the model can look at. */
        if (n >= 9 && strncasecmp(value + 6, "svg", 3) == 0)
            return CT_NOT_IMAGE;
        return CT_IMAGE;
    }
    if ((n == 24 && strncasecmp(value, "application/octet-stream", 24) == 0) ||
        (n == 19 && strncasecmp(value, "binary/octet-stream", 19) == 0) ||
        (n == 18 && strncasecmp(value, "application/binary", 18) == 0))
        return CT_AMBIGUOUS;
    return CT_NOT_IMAGE;
}

img_kind_t image_sniff(const uint8_t *d, size_t n)
{
    static const uint8_t png[8] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };

    if (n >= 3 && d[0] == 0xFF && d[1] == 0xD8 && d[2] == 0xFF)
        return IMG_JPEG;
    if (n >= 8 && memcmp(d, png, 8) == 0)
        return IMG_PNG;
    if (n >= 6 && (memcmp(d, "GIF87a", 6) == 0 || memcmp(d, "GIF89a", 6) == 0))
        return IMG_OTHER;
    if (n >= 12 && memcmp(d, "RIFF", 4) == 0 && memcmp(d + 8, "WEBP", 4) == 0)
        return IMG_WEBP;
    if (n >= 12 && memcmp(d + 4, "ftyp", 4) == 0 &&
        (memcmp(d + 8, "avif", 4) == 0 || memcmp(d + 8, "avis", 4) == 0 ||
         memcmp(d + 8, "heic", 4) == 0 || memcmp(d + 8, "heix", 4) == 0 ||
         memcmp(d + 8, "mif1", 4) == 0))
        return IMG_OTHER;
    if (n >= 2 && d[0] == 'B' && d[1] == 'M')
        return IMG_OTHER;
    return IMG_UNKNOWN;
}

const char *image_kind_name(img_kind_t k)
{
    switch (k) {
    case IMG_JPEG:  return "jpeg";
    case IMG_PNG:   return "png";
    case IMG_WEBP:  return "webp";
    case IMG_OTHER: return "other";
    default:        return "unknown";
    }
}

int accept_drop_avif(const char *v, size_t n, char *out, size_t cap, size_t *outn)
{
    size_t o = 0;
    int dropped = 0;

    if (cap < n + 4)
        return -1;
    size_t i = 0;
    while (i <= n) {
        size_t e = i;
        while (e < n && v[e] != ',')
            e++;
        /* entry is v[i..e): find its media type, ignoring blanks and parameters */
        size_t s = i;
        while (s < e && (v[s] == ' ' || v[s] == '\t'))
            s++;
        size_t t = s;
        while (t < e && v[t] != ';' && v[t] != ' ' && v[t] != '\t')
            t++;
        if (t - s == 10 && strncasecmp(v + s, "image/avif", 10) == 0) {
            dropped = 1;
        } else if (t > s) {
            if (o)
                out[o++] = ',';
            size_t from = o ? i : s;             /* no leading blank on the first entry */
            memcpy(out + o, v + from, e - from);
            o += e - from;
        }
        i = e + 1;
    }
    if (!dropped)
        return 0;
    if (o == 0) {                                /* the client only accepted AVIF */
        memcpy(out, "*/*", 3);
        o = 3;
    }
    *outn = o;
    return 1;
}
