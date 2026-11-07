/* Table test for accept_drop_avif(): only the AVIF entry goes, nothing else moves. */
#include <stdio.h>
#include <string.h>

#include "image_detect.h"

static int failed;

static void expect(const char *in, const char *want /* NULL = unchanged */)
{
    char out[256];
    size_t n = 0;
    int r = accept_drop_avif(in, strlen(in), out, sizeof out, &n);
    int ok = want ? (r == 1 && n == strlen(want) && memcmp(out, want, n) == 0) : r == 0;
    printf("%s Accept: %s  ->  %s\n", ok ? "PASS" : "FAIL", in, want ? want : "(unchanged)");
    if (!ok) {
        failed = 1;
        if (r == 1)
            printf("     got: %.*s\n", (int)n, out);
    }
}

int main(void)
{
    /* what Chrome sends for images and for documents */
    expect("image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
           "image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8");
    expect("text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
           "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8");
    expect("image/webp,image/avif", "image/webp");
    expect("image/avif;q=0.9,image/webp", "image/webp");
    expect("IMAGE/AVIF, image/webp, */*", "image/webp, */*");
    expect("image/webp, image/avif ,*/*;q=0.5", "image/webp,*/*;q=0.5");
    expect("image/avif", "*/*");
    expect("image/avif,image/avif", "*/*");
    /* untouched */
    expect("*/*", NULL);
    expect("image/webp,image/png,*/*;q=0.8", NULL);
    expect("image/avifx,text/html", NULL);          /* a different type that merely starts alike */
    expect("text/avif", NULL);
    expect("", NULL);
    return failed;
}
