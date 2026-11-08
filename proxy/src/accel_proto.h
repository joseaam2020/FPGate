#ifndef FPGATE_ACCEL_PROTO_H
#define FPGATE_ACCEL_PROTO_H

#include <stdint.h>

/*
 * Wire format between fpgate-proxy and fpgate-accel (Unix stream socket).
 * Both processes run on the same ARM HPS, so fields are little-endian and
 * the float is IEEE 754 binary32. Every message starts with a fixed 24-byte
 * header; a classify request is followed by `payload_len` bytes of pixels.
 *
 * The proxy sends decoded, resized RGB pixels. Normalisation, quantisation
 * and everything model-specific belong to fpgate-accel. The accelerator
 * reports a score only: the threshold and the failure policy are the proxy's.
 */

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the accel protocol assumes a little-endian host"
#endif

#define ACCEL_PROTO_VERSION 1
#define ACCEL_REQ_MAGIC  "FPGA"
#define ACCEL_RESP_MAGIC "FPGR"
#define ACCEL_MAX_PAYLOAD (16u * 1024 * 1024)

enum accel_msg_type {
    ACCEL_MSG_CLASSIFY = 1,
    ACCEL_MSG_STATUS   = 2,
};

enum accel_pixfmt {
    ACCEL_PIXFMT_RGB8_HWC = 0,      /* 3 channels, uint8, row-major, interleaved */
};

enum accel_status {
    ACCEL_OK          = 0,
    ACCEL_BUSY        = 1,
    ACCEL_HW_ERROR    = 2,
    ACCEL_UNSUPPORTED = 3,          /* version, size or pixel format not accepted */
    ACCEL_DEGRADED    = 4,
};

typedef struct __attribute__((packed)) {
    char     magic[4];              /* "FPGA" */
    uint8_t  version;
    uint8_t  type;                  /* enum accel_msg_type */
    uint8_t  channels;
    uint8_t  pixfmt;                /* enum accel_pixfmt */
    uint32_t request_id;            /* echoed in the response */
    uint16_t width;
    uint16_t height;
    uint32_t payload_len;           /* width * height * channels for classify */
    uint32_t reserved;
} accel_req_hdr_t;

typedef struct __attribute__((packed)) {
    char     magic[4];              /* "FPGR" */
    uint8_t  version;
    uint8_t  type;
    uint8_t  status;                /* enum accel_status; score is void unless OK */
    uint8_t  reserved;
    uint32_t request_id;
    float    score_nsfw;            /* 0..1 */
    uint32_t hw_cycles;
    uint32_t latency_us;
} accel_resp_hdr_t;

_Static_assert(sizeof(accel_req_hdr_t) == 24, "request header must be 24 bytes");
_Static_assert(sizeof(accel_resp_hdr_t) == 24, "response header must be 24 bytes");

#endif
