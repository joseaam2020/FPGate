#define _GNU_SOURCE
#include "settings.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "inspect.h"
#include "substitute.h"

#define SEC_CLASS  "clasificacion"
#define SEC_IMAGES "imagenes"
#define PATH_MAX_LEN 400

static struct {
    char path[PATH_MAX_LEN];        /* config file; empty when there is none */
    char dir[PATH_MAX_LEN];
    char substitute[PATH_MAX_LEN];  /* custom stand-in in use; empty for the default */
} G;

/* ---- the TOML subset: [section], key = number | "string" | true | false, # comments --- */

typedef struct {
    int has_threshold, has_substitute;
    double threshold;
    char substitute[PATH_MAX_LEN];
} fileconf_t;

static char *trim(char *s)
{
    while (isspace((unsigned char)*s))
        s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1]))
        *--e = 0;
    return s;
}

/* Reads `"text"` (with \\ and \" escapes) into dst. Returns 0, or -1 if malformed. */
static int parse_string(const char *v, char *dst, size_t cap)
{
    size_t n = 0;
    if (*v++ != '"')
        return -1;
    while (*v && *v != '"') {
        if (*v == '\\' && (v[1] == '\\' || v[1] == '"'))
            v++;
        if (n + 1 >= cap)
            return -1;
        dst[n++] = *v++;
    }
    if (*v != '"')
        return -1;
    for (v++; *v; v++)
        if (!isspace((unsigned char)*v) && *v != '#')
            return -1;
    dst[n] = 0;
    return 0;
}

static int parse_file(const char *path, fileconf_t *out, char *err, size_t err_len)
{
    memset(out, 0, sizeof *out);
    FILE *f = fopen(path, "r");
    if (!f) {
        if (errno == ENOENT)
            return 0;
        snprintf(err, err_len, "cannot read %s", path);
        return -1;
    }
    char line[600], section[48] = "";
    for (int ln = 1; fgets(line, sizeof line, f); ln++) {
        char *s = trim(line);
        if (!*s || *s == '#')
            continue;
        if (*s == '[') {
            char *e = strchr(s, ']');
            if (!e) {
                snprintf(err, err_len, "%s line %d: unclosed '['", path, ln);
                fclose(f);
                return -1;
            }
            *e = 0;
            snprintf(section, sizeof section, "%s", trim(s + 1));
            continue;
        }
        char *eq = strchr(s, '=');
        if (!eq) {
            snprintf(err, err_len, "%s line %d: expected key = value", path, ln);
            fclose(f);
            return -1;
        }
        *eq = 0;
        char *key = trim(s), *val = trim(eq + 1);
        if (!strcmp(section, SEC_CLASS) && !strcmp(key, "umbral")) {
            char *end;
            double v = strtod(val, &end);
            while (isspace((unsigned char)*end))
                end++;
            if (end == val || (*end && *end != '#')) {
                snprintf(err, err_len, "%s line %d: umbral must be a number", path, ln);
                fclose(f);
                return -1;
            }
            out->threshold = v;
            out->has_threshold = 1;
        } else if (!strcmp(section, SEC_IMAGES) && !strcmp(key, "sustitucion")) {
            if (parse_string(val, out->substitute, sizeof out->substitute) != 0) {
                snprintf(err, err_len, "%s line %d: sustitucion must be a quoted path", path, ln);
                fclose(f);
                return -1;
            }
            out->has_substitute = 1;
        }
        /* any other key belongs to a part that does not read it yet: left alone */
    }
    fclose(f);
    return 0;
}

/*
 * Sets `key = value_text` inside [section] of the file, keeping every other line
 * as it is. Written to a temporary file and renamed, so a crash leaves either the
 * old file or the new one.
 */
static int upsert(const char *path, const char *section, const char *key, const char *value_text,
                  char *err, size_t err_len)
{
    char tmp[PATH_MAX_LEN + 8], line[600], cur[48] = "";
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *in = fopen(path, "r"), *out = fopen(tmp, "w");
    if (!out) {
        snprintf(err, err_len, "cannot write %s", tmp);
        if (in)
            fclose(in);
        return -1;
    }
    int in_target = 0, done = 0, seen_section = 0;
    while (in && fgets(line, sizeof line, in)) {
        char copy[600], *s;
        snprintf(copy, sizeof copy, "%s", line);
        s = trim(copy);
        if (*s == '[') {
            if (in_target && !done) {
                fprintf(out, "%s = %s\n", key, value_text);
                done = 1;
            }
            char *e = strchr(s, ']');
            if (e)
                *e = 0;
            snprintf(cur, sizeof cur, "%s", trim(s + 1));
            in_target = !strcmp(cur, section);
            seen_section |= in_target;
        } else if (in_target && !done && *s && *s != '#') {
            char *eq = strchr(s, '=');
            if (eq) {
                *eq = 0;
                if (!strcmp(trim(s), key)) {
                    fprintf(out, "%s = %s\n", key, value_text);
                    done = 1;
                    continue;
                }
            }
        }
        fputs(line, out);
    }
    if (in)
        fclose(in);
    if (!done) {
        if (!seen_section)
            fprintf(out, "%s[%s]\n", ftell(out) > 0 ? "\n" : "", section);
        fprintf(out, "%s = %s\n", key, value_text);
    }
    int ok = fflush(out) == 0 && fsync(fileno(out)) == 0;
    ok = fclose(out) == 0 && ok;
    if (!ok || rename(tmp, path) != 0) {
        unlink(tmp);
        snprintf(err, err_len, "cannot save %s", path);
        return -1;
    }
    return 0;
}

/* ---- applying ----------------------------------------------------------------------------- */

static int threshold_ok(double v) { return isfinite(v) && v >= 0.0 && v <= 1.0; }

static void set_dir_from(const char *path)
{
    snprintf(G.dir, sizeof G.dir, "%s", path);
    char *slash = strrchr(G.dir, '/');
    if (slash && slash != G.dir)
        *slash = 0;
    else if (slash)
        slash[1] = 0;
    else
        snprintf(G.dir, sizeof G.dir, ".");
}

static int apply(const fileconf_t *c, char *err, size_t err_len)
{
    if (c->has_threshold && !threshold_ok(c->threshold)) {
        snprintf(err, err_len, "umbral must be between 0 and 1 (the file says %g)", c->threshold);
        return -1;
    }
    if (c->has_substitute && substitute_load(c->substitute, err, err_len) != 0)
        return -1;
    if (c->has_threshold)
        inspect_set_threshold((float)c->threshold);
    if (c->has_substitute)
        snprintf(G.substitute, sizeof G.substitute, "%s", c->substitute);
    return 0;
}

int settings_init(const char *path, char *err, size_t err_len)
{
    if (!path) {
        snprintf(G.dir, sizeof G.dir, "/tmp");
        return 0;
    }
    snprintf(G.path, sizeof G.path, "%s", path);
    set_dir_from(path);
    fileconf_t c;
    if (parse_file(path, &c, err, err_len) != 0)
        return -1;
    return apply(&c, err, err_len);
}

int settings_reload(char *err, size_t err_len)
{
    if (!G.path[0]) {
        snprintf(err, err_len, "no configuration file in use");
        return -1;
    }
    fileconf_t c;
    if (parse_file(G.path, &c, err, err_len) != 0)
        return -1;
    return apply(&c, err, err_len);
}

int settings_set_threshold(double value, char *err, size_t err_len)
{
    if (!threshold_ok(value)) {
        snprintf(err, err_len, "the threshold must be a number between 0 and 1");
        return -1;
    }
    if (G.path[0]) {
        char text[32];
        snprintf(text, sizeof text, "%.4g", value);
        if (upsert(G.path, SEC_CLASS, "umbral", text, err, err_len) != 0)
            return -1;
    }
    inspect_set_threshold((float)value);
    return 0;
}

int settings_set_substitute(const char *upload_path, char *err, size_t err_len)
{
    if (substitute_load(upload_path, err, err_len) != 0) {
        unlink(upload_path);
        return -1;
    }
    const char *type = substitute_custom_type();
    const char *ext = !strcmp(type, "image/jpeg") ? "jpg" : !strcmp(type, "image/webp") ? "webp" : "png";
    char final[PATH_MAX_LEN];
    if (snprintf(final, sizeof final, "%s/substitute.%s", G.dir, ext) >= (int)sizeof final) {
        snprintf(err, err_len, "the configuration directory path is too long");
        unlink(upload_path);
        return -1;
    }
    if (G.path[0]) {
        char quoted[PATH_MAX_LEN + 8];
        snprintf(quoted, sizeof quoted, "\"%s\"", final);
        if (rename(upload_path, final) != 0) {
            snprintf(err, err_len, "cannot store the image in %s", G.dir);
            unlink(upload_path);
            return -1;
        }
        if (upsert(G.path, SEC_IMAGES, "sustitucion", quoted, err, err_len) != 0)
            return -1;
        if (G.substitute[0] && strcmp(G.substitute, final) != 0)
            unlink(G.substitute);
        snprintf(G.substitute, sizeof G.substitute, "%s", final);
    } else {
        unlink(upload_path);        /* already in memory; nothing to keep */
        G.substitute[0] = 0;
    }
    return 0;
}

int settings_persistent(void) { return G.path[0] != 0; }
const char *settings_dir(void) { return G.dir[0] ? G.dir : "/tmp"; }
const char *settings_substitute_path(void) { return G.substitute[0] ? G.substitute : NULL; }
