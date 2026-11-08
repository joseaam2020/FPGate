/*
 * The configuration file: parser, validation and the in-place rewrite of one key.
 * The image loader and the proxy core are replaced by stubs (no file is opened by
 * name from fuzzed text).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "settings.h"

int substitute_load(const char *path, char *err, size_t n) { (void)path; snprintf(err, n, "stub"); return -1; }
const char *substitute_custom_type(void) { return NULL; }
void inspect_set_threshold(float t) { (void)t; }

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    static char path[64];
    if (!path[0])
        snprintf(path, sizeof path, "/tmp/fuzz_toml_%d.toml", (int)getpid());
    FILE *f = fopen(path, "wb");
    if (!f)
        return 0;
    fwrite(data, 1, size, f);
    fclose(f);

    char err[200];
    settings_init(path, err, sizeof err);
    settings_reload(err, sizeof err);
    settings_set_threshold(0.5, err, sizeof err);       /* rewrites the file around the fuzzed text */
    return 0;
}
