/* The administrator credentials file. */
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#include "auth.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    static char path[64];
    if (!path[0])
        snprintf(path, sizeof path, "/tmp/fuzz_creds_%d", (int)getpid());
    FILE *f = fopen(path, "wb");
    if (!f)
        return 0;
    fwrite(data, 1, size, f);
    fclose(f);
    char err[200];
    (void)auth_load(path, err, sizeof err);
    return 0;
}
