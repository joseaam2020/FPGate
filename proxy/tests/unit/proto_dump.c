/*
 * Writes one request header and one response header with fixed values to
 * stdout. The mock-accel test compares these bytes with what Python's struct
 * module produces, so both sides of the wire format are checked against each
 * other.
 */
#include <stdio.h>
#include <string.h>

#include "accel_proto.h"

int main(void)
{
    accel_req_hdr_t rq;
    accel_resp_hdr_t rs;

    memset(&rq, 0, sizeof rq);
    memcpy(rq.magic, ACCEL_REQ_MAGIC, 4);
    rq.version = ACCEL_PROTO_VERSION;
    rq.type = ACCEL_MSG_CLASSIFY;
    rq.channels = 3;
    rq.pixfmt = ACCEL_PIXFMT_RGB8_HWC;
    rq.request_id = 0x01020304u;
    rq.width = 224;
    rq.height = 336;
    rq.payload_len = 224u * 336u * 3u;

    memset(&rs, 0, sizeof rs);
    memcpy(rs.magic, ACCEL_RESP_MAGIC, 4);
    rs.version = ACCEL_PROTO_VERSION;
    rs.type = ACCEL_MSG_CLASSIFY;
    rs.status = ACCEL_DEGRADED;
    rs.request_id = 0x0A0B0C0Du;
    rs.score_nsfw = 0.5f;
    rs.hw_cycles = 123456789u;
    rs.latency_us = 4200000u;

    fwrite(&rq, sizeof rq, 1, stdout);
    fwrite(&rs, sizeof rs, 1, stdout);
    return 0;
}
