#ifndef FPGATE_SETTINGS_H
#define FPGATE_SETTINGS_H

#include <stddef.h>

/*
 * Settings an administrator can change while the proxy runs (RF-09, RI-02),
 * kept in a TOML file:
 *
 *     [clasificacion]
 *     umbral = 0.50
 *
 *     [imagenes]
 *     sustitucion = "/etc/fpgate/substitute.png"
 *
 * Values found in the file override the --threshold and --substitute flags.
 * A change is validated first; if it is not valid nothing changes and the
 * error says why (RU-06). Without a file (path NULL) changes still apply but
 * are lost on restart.
 *
 * Every function runs on the event loop thread, so a change affects the
 * classifications that start after it.
 */

/* Loads and applies the file if it exists. 0, or -1 with a message in err. */
int settings_init(const char *path, char *err, size_t err_len);
/* Re-reads the file (SIGHUP). On error the settings in force are kept. */
int settings_reload(char *err, size_t err_len);

int settings_set_threshold(double value, char *err, size_t err_len);
/*
 * Adopts the image stored at `upload_path` (written by the caller inside
 * settings_dir()) as the stand-in: validates it, moves it next to the config
 * file and records it. The upload file is consumed either way.
 */
int settings_set_substitute(const char *upload_path, char *err, size_t err_len);

int settings_persistent(void);
/* Directory for uploads and the stored stand-in. */
const char *settings_dir(void);
/* Path of the custom stand-in, or NULL when the generated grey PNG is used. */
const char *settings_substitute_path(void);

#endif
