# fpgate-proxy

Proxy HTTP/HTTPS que revisa las imágenes que pasan por él, pregunta a `fpgate-accel` si son NSFW y sustituye las que superan el umbral. Corre en la placa (HPS, ARMv7) y se administra desde un navegador.

Esta guía instala el proxy en la placa y deja funcionando la interfaz de administración. Los detalles de diseño y las decisiones están en [DECISIONES.md](DECISIONES.md).

## Resumen: de cero a ver el panel

Todo lo de esta lista se explica con detalle en las secciones siguientes.

1. **PC:** compilar el proxy y, si todavía no hay `fpgate-accel`, el acelerador simulado (sección 1).
2. **PC:** generar la CA (sección 2).
3. **Placa:** copiar los archivos y crear la contraseña de `admin` (sección 3). Sin ese archivo **no arranca el panel**.
4. **Placa:** arrancar el acelerador simulado y después el proxy con `--admin-credentials` (sección 4). Al arrancar debe aparecer la línea `admin interface: https://<this address>:8443`. Si no aparece, el panel no está corriendo: ver «Si el panel no abre» (sección 7).
5. **PC de prueba:** instalar el certificado y abrir Chrome con el proxy (sección 5).
6. **PC de prueba:** entrar a **https://fpgate.admin** (sección 6).

## 1. Compilar (en la PC)

Todo se compila dentro de Docker; no hace falta instalar nada más.

```sh
cd proxy
docker compose build                                    # una sola vez
docker compose run --rm dev make arm                    # build/arm/fpgate-proxy (estático, musl)
docker compose run --rm dev make build/arm/mock_accel   # acelerador simulado, mientras no exista fpgate-accel
docker compose run --rm dev make test                   # opcional: toda la batería de pruebas
```

El binario ARM ya lleva dentro las tres páginas de administración (`web/`). No hay que copiar nada más que el ejecutable.

## 2. Generar la CA y la contraseña (en la PC, una sola vez)

La CA es la autoridad con la que el proxy firma los certificados de los sitios HTTPS y de la propia interfaz. **La clave `ca.key` nunca sale de la placa**; solo `ca.crt` se reparte.

```sh
mkdir -p certs/ca && cd proxy
export FPGATE_CA_PASS='una contraseña larga para la clave de la CA'
docker compose run --rm -e FPGATE_CA_PASS dev sh tools/gen_ca.sh certs/ca 365
docker compose run --rm dev sh tools/fetch_ca_bundle.sh certs/ca-bundle.pem   # CAs de confianza para los servidores reales
printf '%s\n' "$FPGATE_CA_PASS" > certs/ca.pass
```

La generación se hace en la PC porque el OpenSSL de la placa (1.0.1) es demasiado viejo para `gen_ca.sh`.

## 3. Copiar a la placa

```sh
ssh root@<ip-de-la-placa> 'mkdir -p /etc/fpgate && chmod 700 /etc/fpgate'
ssh root@<ip-de-la-placa> 'killall fpgate-proxy; killall gatord'          # gatord ocupa el 8080
scp -O build/arm/fpgate-proxy build/arm/mock_accel root@<ip-de-la-placa>:/usr/local/bin/
scp -O certs/ca/ca.crt certs/ca/ca.key certs/ca.pass certs/ca-bundle.pem \
    root@<ip-de-la-placa>:/etc/fpgate/
ssh root@<ip-de-la-placa> 'chmod 600 /etc/fpgate/*.key /etc/fpgate/*.pass'
```

Contraseña de la persona administradora (usuario `admin`; se pide sin mostrarla, mínimo 8 caracteres). Se crea en la placa, así no viaja por la red:

```sh
ssh -t root@<ip-de-la-placa> 'fpgate-proxy --set-admin-password /etc/fpgate/admin.pass'
```

No existe una contraseña por defecto: sin este archivo la interfaz de administración no arranca.

La hora de la placa debe ser correcta (`date`, o NTP); si no, los certificados no validan.

## 4. Arrancar

Comprobar primero que la placa tiene el binario nuevo (debe decir `0.5.0` o más):

```sh
fpgate-proxy -v
```

Si todavía no existe `fpgate-accel`, arrancar el acelerador simulado en otra terminal de la placa (**no usa el acelerador**: califica cada imagen por su brillo medio, así las claras se bloquean):

```sh
mock_accel /run/fpgate-accel.sock --mean --latency-ms 200
```

Sin nada escuchando en el socket, el proxy entra en modo degradado y bloquea todas las imágenes (`reason=accel_down`).

Después, el proxy:

```sh
fpgate-proxy -p 8080 \
    -C /etc/fpgate/ca.crt -K /etc/fpgate/ca.key -P /etc/fpgate/ca.pass -B /etc/fpgate/ca-bundle.pem \
    --accel /run/fpgate-accel.sock \
    --block-log /etc/fpgate/blocks.db \
    --config /etc/fpgate/fpgate.toml \
    --admin-credentials /etc/fpgate/admin.pass
```

| Opción | Para qué sirve |
|---|---|
| `-p` | Puerto del proxy (8080). |
| `-C -K -P -B` | Habilita HTTPS: certificado y clave de la CA, archivo con su contraseña y CAs de confianza. |
| `--accel` | Socket de `fpgate-accel`. Sin esto las imágenes solo se analizan y se registran. |
| `--block-log` | Bitácora SQLite de bloqueos (dominio, score, umbral, motivo; nunca URL ni píxeles). |
| `--config` | Archivo TOML con el umbral y la imagen de sustitución. La página de administración lo escribe; `kill -HUP` lo vuelve a leer. |
| `--admin-credentials` | **Activa el panel de administración** (requiere `-C -K`). Sin esta opción no hay panel. Puerto directo `8443` (cambiar con `--admin-port`). |

Al arrancar, el log debe incluir estas dos líneas; si falta la segunda, el panel no está corriendo:

```
tls: CA valid, HTTPS interception enabled
admin interface: https://<this address>:8443
```

`/etc/fpgate/fpgate.toml` no hace falta crearlo: lo crea la página la primera vez que se guarda algo. Si se escribe a mano:

```toml
[clasificacion]
umbral = 0.50          # entre 0 y 1; un valor menor filtra más

[imagenes]
sustitucion = "/etc/fpgate/substitute.png"
```

Un valor inválido en el archivo se rechaza al recargar y el proxy conserva la configuración anterior (lo dice el log).

## 5. Instalar el certificado en el dispositivo de prueba

La persona que va a usar el dispositivo no necesita la contraseña de administración:

1. Abrir Chrome conectado al proxy (solo esa ventana usa FPGate):
   `chrome.exe --proxy-server="<ip-de-la-placa>:8080"`
2. Entrar a **http://fpgate.setup**. Es la guía paso a paso, con el botón para descargar `fpgate-ca.crt`.
3. Instalar el certificado en *Entidades de certificación raíz de confianza* y volver a abrir Chrome con el mismo comando.

La misma página está en `https://fpgate.admin/certificado`. Instalar el certificado solo en el dispositivo de prueba.

## 6. Administración

Con Chrome abierto a través del proxy (paso 5), entrar a **https://fpgate.admin** e iniciar sesión como `admin`. El proxy entrega el panel él mismo, con un certificado para ese nombre, así que funciona aunque la IP de la placa cambie. (Equivale a `http://fpgate.setup`, pero cifrado.)

También se puede entrar directamente, sin pasar por el proxy, a `https://<nombre-o-ip-de-la-placa>:8443`; el certificado de ese puerto cubre el nombre de la placa (`socfpga`), `socfpga.local` y la IP con la que se entra.

- **Umbral:** el cambio se aplica a las imágenes que lleguen a partir de ese momento, sin reiniciar.
- **Imagen de sustitución:** JPG, PNG o WebP de hasta 2 MB; se valida antes de aceptarla.
- **Estadísticas:** latencia del acelerador, throughput y estado (`normal`, `degradado`, `configuración inicial`).
- **Bitácora:** bloqueos paginados.

Tras cinco contraseñas incorrectas seguidas los inicios de sesión se bloquean 30 segundos. Las sesiones caducan a los 30 minutos sin actividad.

Para cambiar la contraseña: repetir `--set-admin-password` y reiniciar el proxy.

## 7. Si el panel no abre

Revisar en este orden:

| Síntoma | Causa y solución |
|---|---|
| El log de arranque **no** muestra `admin interface: https://...:8443` | No se pasó `--admin-credentials`, o falta el archivo: crearlo con `fpgate-proxy --set-admin-password /etc/fpgate/admin.pass`. Si el log dice `admin: the interface needs the CA`, faltan `-C`/`-K`. |
| `--admin-credentials: cannot open ...` y el proxy no arranca | La ruta del archivo de credenciales es incorrecta. |
| `bind/listen: Address already in use` | Quedó otro proxy (o `gatord`, que ocupa el 8080): `killall fpgate-proxy; killall gatord`. |
| `scp`: `Text file busy` | El proxy sigue corriendo: pararlo antes de copiar el binario nuevo. |
| `fpgate-proxy -v` muestra un número menor que 0.5.0 | La placa tiene un binario viejo (sin panel): copiar `build/arm/fpgate-proxy` otra vez. |
| Chrome dice «Tu conexión no es privada» | El certificado de FPGate no está instalado en esa PC (sección 5), o Chrome no se abrió con `--proxy-server`. Si el equipo tiene antivirus o control parental que revisa HTTPS, mostrará su propio error. |
| `https://fpgate.admin` no resuelve o da error de red | Chrome no está usando el proxy: ese nombre solo existe a través de él. Abrir Chrome con `chrome.exe --proxy-server="<ip-de-la-placa>:8080"`. |
| «FPGate: no se pudo contactar al servidor» | Se pidió una dirección que el proxy tomó como un sitio normal, por ejemplo `socfpga:8443` sin `https://`. Usar `https://fpgate.admin`. |
| Muchos errores de certificado en todos los sitios | La hora de la placa está mal: `date`, o `date -s "AAAA-MM-DD HH:MM:SS"`. |
| «Demasiados intentos. Espera N segundos» | Cinco contraseñas incorrectas seguidas: esperar 30 segundos. |
| Todas las imágenes salen bloqueadas | No hay acelerador escuchando en `--accel` (`reason=accel_down` en el log): arrancar `mock_accel`. |

El log del proxy muestra cada conexión (`CONNECT fpgate.admin:443`) y cada decisión sobre imágenes (`VERDICT ... threshold=...`); es lo primero que hay que mirar.
