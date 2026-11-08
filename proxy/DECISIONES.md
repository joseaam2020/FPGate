# DECISIONES.md — fpgate-proxy (DES-3)

Registro de decisiones. Formato: contexto, decision, estado.

## Hallazgos de la placa (2026-11-01)

| Dato | Valor | Consecuencia |
|---|---|---|
| Kernel | 4.5.0 armv7l (2017) | Sin problemas para epoll/accept4 |
| libc | glibc 2.20 | No existe `explicit_bzero` (glibc >= 2.25): usar implementacion propia con barrera de compilador |
| Compilador | gcc 4.9.3 (Linaro) | Se cross-compila en Docker; la placa no compila |
| OpenSSL | 1.0.1p (2015) | Sin TLS 1.3 y obsoleta: **no se usa**; TLS se enlazara estaticamente |
| ABI | hard-float (`arm-linux-gnueabihf`) | Verificado: el binario estatico corre en la placa |
| Home de root | `/home/root` (no `/root`) | Ruta de despliegue |
| Red | eth0 192.168.50.2/24 directo al PC (192.168.50.1) | Configurado a mano, se pierde al reiniciar |

## Flujo de desarrollo

- Se compila todo en Docker (`proxy/docker-compose.yml`), targets `native`, `asan`, `ubsan`, `arm`, `test`.
- El binario ARM es **estatico** para no depender de la glibc de la placa.
- Despliegue: `scp -O build/arm/fpgate-proxy root@192.168.50.2:/home/root/fpgate-proxy` (`-O` y ruta con nombre completo: el scp remoto es antiguo).

## #3, #4, #5 Contrato `classify` — CERRADAS (2026-11-02)

Definicion en `src/accel_proto.h`; el mock (`tests/mock_accel/mock_accel.py`) y `tests/mock_accel/test_mock_accel.py` verifican byte a byte que C y Python coinciden.

**Reparto de responsabilidades**
- El proxy decodifica la imagen, la redimensiona a la resolucion que pida el acelerador y envia **RGB uint8 (HWC)**.
- `fpgate-accel` normaliza, cuantiza a INT8, coloca los datos en DDR3, maneja CSR/DMA y las operaciones del HPS. Todo lo especifico del modelo (media/desviacion, escalas, clases que forman el score) vive ahi, junto al codigo de entrenamiento y cuantizacion: la poda y la cuantizacion en curso no tocan el proxy.
- El acelerador **no decide**: devuelve `score_nsfw`. El umbral (RF-09) y la politica ante fallas (RF-14) son del proxy.
- Resolucion (#4): no esta fijada en el proxy; viaja en la cabecera y se configura. Valor por defecto del mock: 224.
- Normalizacion (#5): sale del proxy. Lo que queda en el proxy es el **metodo de interpolacion del redimensionado**, que debe coincidir con el entrenamiento; no sera identico byte a byte a PIL, la prueba de DES-3.2 usara una tolerancia documentada.

**Mensajes** (socket Unix, little-endian, cabecera fija de 24 bytes, version 1)

| Peticion `FPGA` | | Respuesta `FPGR` | |
|---|---|---|---|
| magic | 4 | magic | 4 |
| version, tipo (1 classify, 2 status), canales, formato (0 = RGB8 HWC) | 4 x u8 | version, tipo, estado, reservado | 4 x u8 |
| request_id | u32 | request_id | u32 |
| ancho, alto | 2 x u16 | score_nsfw | f32 |
| payload_len | u32 | hw_cycles | u32 |
| reservado | u32 | latency_us | u32 |

Estados: `OK`, `BUSY`, `HW_ERROR`, `UNSUPPORTED` (version, tamano o formato no aceptado), `DEGRADED`. Si el estado no es OK el score no es valido y el proxy aplica la politica por defecto.

**Por que `request_id`**: con latencias de 3-7 s, si el proxy vence el timeout y reutiliza la conexion, una respuesta tardia podria entregarse como si fuera de la imagen siguiente. El proxy descarta respuestas cuyo id no coincide.
**Por que `status`**: permite probar el acelerador para salir del modo degradado sin arriesgar una imagen real.

**Mock**: procesa una imagen a la vez (como el motor GEMM); score fijo o `score_from_pixels` (imagen blanca = 1.0), latencia configurable, fallas `error` / `close` / `hang`, configuracion JSON releida en cada peticion (se puede cambiar con el proxy corriendo) y bitacora con id, tamano y sha256 de los pixeles (nunca los pixeles) para verificar RF-03.

> **Nota para el desarrollo de `fpgate-accel`**: el mock es la referencia ejecutable del contrato y `src/accel_proto.h` es la fuente de verdad del formato. Si el formato cambia, se cambia primero en `accel_proto.h` y luego en el mock y en sus pruebas. Al empezar `fpgate-accel` se puede usar `tests/mock_accel/test_mock_accel.py` como especificacion de comportamiento. Hoy ese test arranca el mock por su cuenta; para ejecutarlo contra el servidor real habria que anadirle una opcion `--socket` y omitir las pruebas que dependen de la configuracion del mock (`fail=*`, `score_from_pixels`, latencia). Las comprobaciones reutilizables tal cual son: formato en el cable, eco del `request_id`, rechazo con `UNSUPPORTED` (version, tamano, formato de pixel) y respuesta al mensaje `status`.

## DES-3.2 Extraccion de imagenes (2026-11-02)

**Flujo** (`src/http_stream.c`, `src/inspect.c`, `src/image_detect.c`, `src/image_prep.c`, `src/resample.c`): en HTTP en claro y en el lado descifrado de un tunel interceptado, el proxy analiza el framing HTTP/1.1 de cada respuesta (Content-Length, chunked con extensiones y trailers, cierre de conexion, HEAD/204/304/1xx, keep-alive, pipelining). Todo lo que no es imagen se reenvia de inmediato y sin cambios. Una respuesta 200 con imagen se retiene hasta completarla, se decodifica, se preprocesa y se entrega **identica byte a byte** (hoy; en DES-3.3 el veredicto de `fpgate-accel` decide entre eso y la sustitucion).

**Deteccion**: `Content-Type` image/* (salvo svg) -> se retiene; ausente u `octet-stream` -> se retiene hasta ver 12 bytes y se decide por magic bytes; cualquier otro tipo -> no se mira. **Los magic bytes mandan** sobre la cabecera (un PNG etiquetado image/jpeg se decodifica como PNG). Limite conocido: una imagen mal etiquetada como `text/*` o similar no se detecta.

**Equivalencia con el entrenamiento** (el modelo usa `Image.open().convert("RGB")` + transformacion eval de timm: lado corto a 448 con bicubic + CenterCrop):
- `resample.c` reproduce el `ImagingResample` de Pillow (coeficientes bicubicos a=-0.5 en punto fijo de 22 bits, pasada horizontal y luego vertical, redondeo a 8 bits entre ambas). `tests/unit/test_prep.py` compara contra Pillow: **identico byte a byte** en 36 casos (JPEG baseline/progresivo/4:4:4/4:2:2/gris, PNG RGB/RGBA/L/LA/paleta/1 bit/Adam7; 448 crop y 224 squash; escalado, identidad y aspectos extremos). x86 y ARM+NEON dan el mismo resultado.
- Se imita `convert("RGB")`: el alfa se descarta sin componer, se ignoran gamma/ICC/orientacion EXIF, PNG de 16 bits se trunca a 8.
- Los coeficientes se calculan con `-ffp-contract=off` para que ninguna arquitectura los altere.
- **Por confirmar con el modelo**: que `crop_pct` de la configuracion de timm sea 1.0 y el modo de recorte sea `center` (ejecutar `model/vit_eval.py`, que imprime "Pipeline de transformaciones"). Si no, cambia `-S`/`-Q` o se ajusta `rs_new`.
- `-D` (escalado DCT de libjpeg, 1/2, 1/4, 1/8) acelera la decodificacion de fotos grandes pero **no es identico** al flujo de entrenamiento (diferencia media < 3 niveles en la prueba); queda apagado por defecto hasta medir su efecto en la exactitud.

**Politicas (decisiones #6 y #8, provisionales)**

| Caso | Hoy | Contador |
|---|---|---|
| JPEG/PNG valido | decodificado y preprocesado, entregado | `decoded` |
| JPEG/PNG danado o truncado | entregado, registrado como `corrupt` (no como bloqueo) | `corrupt` |
| WebP (imagen fija) | decodificado y preprocesado igual que JPEG/PNG | `decoded` |
| WebP animado, GIF, AVIF/HEIC, BMP | entregado, `unsupported_format` | `unsupported_format` |
| JPEG CMYK/YCCK o de 12 bits | entregado, `unsupported_format` | `unsupported_format` |
| Mas de 16 MP (`max_pixels`) | entregado, `too_many_pixels` | `too_many_pixels` |
| Mayor que `-L` (8 MiB) | se transmite sin retener | `skipped_oversize` |
| Presupuesto global de retencion (64 MiB) agotado | se transmite sin retener | `skipped_budget` |
| 206 Partial Content | se transmite sin analizar | `skipped_partial` |
| `Content-Encoding` distinto de identity | se transmite sin analizar | `skipped_encoded` |
| Content-Length + Transfer-Encoding, longitudes contradictorias, chunk mal formado | **se cierra la conexion** (un parser que discrepa del navegador permitiria colar imagenes) | - |

**Log de imagenes** (2026-11-04, a raiz de la primera prueba con Chrome real, donde no aparecia ninguna linea `IMG`): cada imagen que el proxy ve deja ahora una linea, tambien las que descarta, con el motivo.
- Analizada: `IMG host=H kind=jpeg src=WxH bytes=N prep=Xms`; fallos: `IMG host=H kind=K bytes=N error=corrupt|unsupported|too_large|unsupported_format|not_an_image`.
- Descartada: `IMG host=H type=image/jpeg skipped=encoded|oversize|budget|partial|busy bytes=N|framing`.
- Con `-V` se agrega ` path=/ruta ref=pagina/origen` (sin esquema ni query string: los tokens y parametros nunca se registran) a cada linea, y una linea `RESP host status type bytes action` por **cada** respuesta, para ver por que algo no se analizo. La ruta y el Referer se emparejan con su respuesta mediante una cola por conexion (pipelining incluido). Sin `-V` no se registra ninguna ruta (RS-05).
- Una respuesta sin `Content-Type` se retiene hasta ver 12 bytes: si no es imagen se libera y con `-V` queda como `action=sniffed-not-image`.

**TLS que no es HTTP** (primera prueba con Chrome real: `mtalk.google.com:5228`, canal push de Google): si lo primero que el cliente envia tras el handshake no se puede leer como una peticion HTTP, el par pasa a opaco (se reenvia sin analizar) en vez de cerrarse con `framing error`. Solo se aplica a los primeros bytes de la conexion, que vienen del cliente; un servidor no puede usarlo para esconder una imagen. Un `framing error` posterior (despues de una peticion HTTP valida) sigue cerrando la conexion.

**Prueba con navegador**: el PC de desarrollo tiene Qustodio (control parental) instalado, que intercepta el HTTPS de los navegadores incluso a traves de un proxy explicito y rechaza la CA de FPGate (Chrome muestra `ERR_CERT_COMMON_NAME_INVALID` con certificados `verifyfail.*` emitidos por `Qustodio (15)`). Las pruebas con navegador se hacen desde otro equipo; `curl`/`openssl` no se ven afectados. Chrome se lanza con `--proxy-server` y un `--user-data-dir` propio para que solo esa instancia use el proxy.

**WebP y AVIF (decision #6, cerrada 2026-11-07)**: la primera prueba con Chrome real contra la portada de Wikipedia mostro que 18 de 20 imagenes de pixeles llegaban como WebP (las URL terminan en `.jpg`/`.png` pero el servidor negocia el formato con la cabecera `Accept`), asi que analizar solo JPEG/PNG dejaba casi toda la web fuera. Decision:
- **Se decodifica WebP** (libwebp 1.5.0, solo el decodificador, estatico, NEON en ARM). Mismas reglas que el resto: alfa descartado sin componer, sin animaciones. Verificado byte a byte contra Pillow (lossy, lossless, con y sin alfa, 448 crop y 224 squash) y entre x86 y ARM+NEON.
- **Se quita `image/avif` de la cabecera `Accept`** de las peticiones (HTTP en claro y tuneles interceptados), para que los servidores que negocian respondan WebP y no AVIF, que no decodificamos (`libavif` + `dav1d` son pesados para la A9). Es lo unico que el proxy modifica de una peticion; se cuenta en `accept_rewritten`. Es una excepcion deliberada a RF-08 (preservar encabezados) y se documentara en la guia de instalacion. Si el cliente solo aceptaba AVIF, `Accept` pasa a `*/*`.
- Limite: un sitio que sirva AVIF sin negociar (o segun el User-Agent) seguira llegando como `unsupported_format`. El log lo mostrara.

**Abierto**: ninguno de estos casos *bloquea* todavia; en DES-3.3 hay que decidir si los formatos que siguen sin decodificar (GIF, AVIF sin negociacion, WebP animado) y las imagenes danadas se bloquean (fail-closed) o se dejan pasar (hueco de contenido). Mi recomendacion: configurable, con "permitir" por defecto para formatos no soportados y "bloquear" para danadas. Decision de producto, no tecnica.

**Decodificacion en hilos de trabajo** (`src/decode_pool.c`, `src/inspect.c`): medido en la placa, decodificar dentro del reactor lo detenia 90-540 ms por imagen y hasta ~1.3 s en el peor caso permitido, contra los 500 ms de RR-03 para el trafico sin imagenes. Ahora:
- La decodificacion y el redimensionado corren en un pool de hilos (`-W n`, por defecto 1; `-W 0` decodifica dentro del reactor, util como referencia). El resultado vuelve al reactor por un `eventfd` registrado en epoll.
- Mientras una imagen esta en un hilo, el par esta "ocupado": el proxy deja de leer del servidor, **los bytes que ya llegaron detras de la imagen se guardan en orden** (backlog) y no se cierra el par aunque el servidor haya cerrado (`eof_pending`). Al terminar se entrega la imagen y se procesa el backlog, que puede iniciar otro trabajo. El parser HTTP se pausa entre mensajes (`hs_stop`) para no reordenar respuestas.
- Si la conexion muere con un trabajo en vuelo, el resultado se descarta: el trabajo es dueno de los bytes retenidos (se vacian con `explicit_bzero`), el hilo nunca toca la conexion.
- Cola de 32 trabajos; si se llena, la imagen pasa sin analizar y se cuenta (`skipped_busy`). Cuando exista DES-3.3, ese caso debe aplicar la politica por defecto, no dejar pasar.
- Los hilos usan 512 KiB de pila (la de musl por defecto es 128 KiB).
- Prueba (`tests/integration/test_workers.sh`, decodificacion con 800 ms de retardo artificial): una peticion de texto concurrente tarda 0.60 s con decodificacion en el reactor y 0.004 s con un hilo de trabajo. Tambien cubre cierre del servidor y aborto del cliente a mitad de la decodificacion, respuestas pipelined en un mismo tunel, 8 y 45 clientes simultaneos y cierre sin fugas.
- Medido en la placa (2026-11-04): 60 peticiones pequenas a http://example.com mientras se descarga la foto de 3840 px (537 ms de decodificacion). Con decodificacion en el reactor (`-W 0`): peor peticion 0.78 s, resto 0.25-0.28 s. Con un hilo (`-W 1`): peor peticion 0.33 s, sin pico. Una sola corrida por modo y por internet (ruido ~50 ms); los logs de ambas instancias confirman que la foto se proceso (`prep=539.0ms` y `prep=537.7ms`). Falta repetirlo varias veces para dar un intervalo.
- **Pendiente de medir en la placa**: el HPS tiene dos nucleos y `fpgate-accel` usa CPU del HPS para las operaciones por fila del modelo; falta medir la carga combinada (decodificacion + inferencia) y fijar prioridades (`nice`) si hace falta.

**Memoria y RS-01**: las imagenes retenidas viven en buffers que se copian y se borran con `explicit_bzero` al crecer y al liberar; lo mismo con filas y pixeles temporales en los decodificadores. Nada se escribe a disco. El redimensionado se hace fila a fila mientras se decodifica, asi que una foto de 24 MP no necesita su RGB completo (solo `alto x 448 x 3` bytes).

**Tiempos medidos en la placa** (2026-11-02, Cortex-A9 con NEON, binario estatico musl, salida 448 crop, `prep_tool`):

| Imagen | Decodificar + redimensionar | Con `-D` |
|---|---|---|
| JPEG 1280x575, 216 KB | 92 ms | - |
| PNG 800x600 RGBA, 219 KB | 118 ms | - |
| JPEG 3840x1725, 1.5 MB | 537 ms | 277 ms |

Por el proxy, JPEG 1280x575 por HTTPS interceptado: `prep=92.0ms` en el log, 200 en 1.06 s totales de punta a punta (incluye handshakes y la descarga desde internet). La resolucion y el codec dominan: extrapolando por pixeles, una foto de 16 MP (el tope) tardaria ~1.3 s exacta y ~0.7 s con `-D`.

**Limitaciones que quedan**
- ~~La decodificacion corre dentro del reactor~~ **Resuelto (2026-11-02)**: ver "Decodificacion en hilos de trabajo".
- Un JPEG progresivo grande necesita sus coeficientes en memoria (limite de libjpeg: 96 MB).
- Sin pipelining >32 solicitudes en vuelo (pasa a opaco sin analizar), sin HTTP/2 (ALPN lo impide).
- El presupuesto de retencion es global y fijo (64 MiB); la concurrencia maxima de imagenes en vuelo no esta limitada de otra forma (decision #8 sigue abierta en este punto).
- Fuzzing del detector, del framing y de los decodificadores: pendiente.

## DES-3.3 Veredicto del acelerador, sustitucion y modo degradado (2026-11-07)

Se activa con `--accel RUTA_DEL_SOCKET`; sin esa opcion el proxy solo analiza y registra, y entrega siempre la imagen original. Probado contra el mock (`tests/integration/test_accel.sh`, 55 comprobaciones bajo ASan/UBSan).

**Flujo**: respuesta de imagen retenida -> hilo de decodificacion -> tensor RGB 448x448 -> cliente del acelerador (`src/accel_client.c`) -> veredicto -> se entrega la original o el reemplazo, y despues lo que haya llegado detras. Una linea `VERDICT` por imagen: `verdict=allow|block reason=... score=... threshold=... accel=Nms prep=Nms total=Nms` (`total` = desde que llega la cabecera de la respuesta hasta la decision; sirve para medir RR-04, separando preprocesamiento y espera del acelerador).

**Decision**: `score_nsfw >= umbral` (por defecto 0.50, `--threshold`) reemplaza. El tensor que recibe el acelerador es byte a byte el que produce `prep_tool` (verificado con el hash que registra el mock).

**Cliente del acelerador**: una peticion en vuelo (el acelerador procesa una imagen a la vez) y una cola FIFO acotada (`--queue-max`, 8 por defecto); si esta llena la imagen se resuelve al instante como `queue_full`, sin esperar. Timeout por peticion desde que se envia (`--accel-timeout-ms`, 10000 por defecto: mayor que la latencia maxima estimada de 3-7 s -- decision #7). Tras un timeout se **cierra la conexion**, asi una respuesta tardia nunca puede confundirse con la de otra imagen; ademas cada respuesta se valida por `request_id`. Cada peticion recibe exactamente una respuesta por su callback (resultado, error, timeout, caido, cola llena, cancelada) y el tensor se borra con `explicit_bzero` en cuanto se envia.

**Modo degradado**: entra ante fallo de conexion, timeout, enlace roto o estado `HW_ERROR`/`DEGRADED`. Mientras dura, las imagenes se resuelven de inmediato como `accel_down` (no se envia nada) y cada `--probe` (2 s) se pregunta `status`; **solo `OK` devuelve a modo normal** (un `BUSY` indica acelerador ocupado o atascado y se sigue degradado). Verificado: error, conexion cortada, acelerador colgado y reinicio del acelerador, con recuperacion automatica.

**Politicas** (RF-14 y decision #6):
- `--on-failure` (acelerador caido/lento/con error, cola de decodificacion llena): **block** por defecto.
- `--on-corrupt` (imagen danada o truncada): **block** por defecto.
- `--on-unexamined` (GIF, WebP animado, AVIF no negociado, mas de 16 MP, no es imagen): **allow** por defecto, para no romper sitios.
- Siguen pasando sin analizar y sin posibilidad de bloqueo (agujero conocido, registrado como `skipped=...`): imagenes mayores que `-L`, `206`, con `Content-Encoding`, o cuando se agota el presupuesto de memoria. Bloquearlas exige descartar el cuerpo restante de la respuesta, que hoy no se retiene.

**Reemplazo** (`src/substitute.c`): una respuesta `200` nueva con una imagen, **nunca** las cabeceras de la original (sin `ETag`, `Last-Modified`, `Content-Encoding`), con `Cache-Control: no-store` y `X-FPGate-Block: <motivo>`, y `Connection: close` si la original cerraba. Por defecto es un PNG gris con borde y cruz, generado a las dimensiones de la original (reducidas a 800 px como maximo, conservando la proporcion; pesa unos cientos de bytes) para que el diseno de la pagina no salte. `--substitute ARCHIVO` pone una imagen propia (JPEG/PNG/WebP, hasta 2 MiB, validada al arrancar), que se envia tal cual (RU-03, RU-04).

**Decision #8 (concurrencia)**: hasta `--queue-max` imagenes esperando mas una en vuelo; el resto se bloquea (o se permite, segun `--on-failure`) al instante. El trafico que no es imagen no se ve afectado (verificado: 0.00x s con la cola llena).

**Pendientes de DES-3.3**: umbral y politicas aun por CLI (pasaran al TOML en DES-3.4); el log de bloqueos va a stderr (la bitacora SQLite es DES-3.4); no hay cache de veredictos por hash ni filtro por tamano minimo (un 16x16 gasta lo mismo que una foto de 12 MP), por decidir con datos de la placa; la espera hasta 10 s por imagen la ve el navegador como una imagen lenta.

## #2 Parser HTTP — CERRADA: picohttpparser

- Vendorizado en `third_party/picohttpparser` (commit en `COMMIT`). Un solo .c/.h, sin dependencias, parsing incremental (`-2` = incompleto).
- No cubre chunked: el decoder de chunked se escribe aparte cuando se necesite reensamblar cuerpos de imagen (DES-3.2).

## #1 Biblioteca TLS — CERRADA: mbedTLS 3.6.4

Medicion en la placa (2026-11-02, 10 corridas, `curl` a https://example.com/, conexion nueva, PC -> placa por router): directo mediana 0.231 s; por el proxy mediana 0.408 s (rango 0.392-0.423 s). Latencia agregada ~177 ms por conexion HTTPS nueva, bajo el limite de 500 ms de RR-03. Limites de la medicion: un solo host cercano (CDN), sin imagenes, una conexion a la vez, sin medir memoria.

- mbedTLS 3.6.4 LTS (sha256 fijado en el Dockerfile), compilada dos veces: nativa y musl-ARM, enlazada estaticamente. Binario ARM de ~0.9 MB con TLS.
- Soporta TLS 1.2/1.3 en ambos roles y escritura de certificados X.509 (generacion de hojas al vuelo). OpenSSL de la placa (1.0.1p) no se usa.
- Verificado en la placa (2026-11-02): `curl --cacert ca.crt -x http://<placa>:8080 https://example.com/` -> 200, hoja emitida por `FPGate CA`, hostname validado. Tiempo total ~0.65 s (hoja nueva) y ~0.61 s (hoja cacheada), con ~56 ms de RTT a internet: generar la hoja no es el cuello de botella; falta separar el costo del proxy (handshake con cliente y con servidor) del costo de la red para evaluar RR-03.
- Pendiente de medir: memoria (RSS) con varias conexiones. Si algo no cumple se reevalua OpenSSL 3.x / wolfSSL.

### Diseno del MITM (src/tls_mitm.c, src/proxy.c)

- Tras el `200 Connection Established` se espera el primer byte del cliente (estado SNIFF): `0x16 0x03` = ClientHello -> se intercepta; cualquier otra cosa, o si el servidor habla primero (SSH, SMTP...), tunel ciego.
- Dos sesiones independientes (cliente<->proxy, proxy<->servidor); los handshakes corren en paralelo y el relay empieza cuando ambos terminan.
- Hoja por **host del CONNECT** (no por SNI): funciona tambien para destinos por IP, que no envian SNI. ECDSA P-256, una sola clave para todas las hojas, SAN (DNS o IP), EKU serverAuth, validez 7 dias (notBefore = ahora - 1 dia), cache LRU de 128 entradas (se guarda el DER; cada sesion parsea su copia).
- Hacia el servidor: TLS >= 1.2, verificacion obligatoria contra el bundle (`-B`, por defecto `/etc/fpgate/ca-bundle.pem`), hostname verificado, ALPN solo `http/1.1`. Hacia el cliente: TLS 1.2 y 1.3; 1.1 recibe alerta `protocol_version`.
- CA: EC P-256, clave cifrada con contrasena (PKCS#8). La contrasena se lee de `-P archivo` o `$FPGATE_CA_PASS`, y se borra de memoria tras cargar la clave. Se avisa si la clave tiene permisos distintos de 600.
- RS-07: la CA se revalida cada 60 s; si esta vencida o aun no es valida, se deja de interceptar y todo HTTPS pasa por tunel ciego.

### Pendientes / riesgos de TLS

- **Reloj de la placa**: arrancaba en 2017. Sin hora correcta las hojas no validan en el cliente y los certificados de los servidores reales fallan la verificacion. El proxy avisa si la hora es anterior a la fecha de compilacion; hay que sincronizar con NTP (o `date -s`) y decidir donde vive la hora en el producto (RTC/NTP).
- Si el cliente rechaza la hoja (CA no instalada, certificate pinning, Firefox con su propio almacen) el handshake falla y la conexion se cierra; aun no hay lista de excepciones para pasar ese host a tunel ciego en el siguiente intento.
- Sin reanudacion de sesiones, sin pool upstream, sin OCSP/CRL de los servidores.
- La clave de la CA y el archivo de contrasena viven en `/etc/fpgate/` con modo 600 (decision de entrega de contrasena aun por cerrar: archivo 600 vs. ingreso interactivo).
- Probado con ASan/UBSan: 15 pruebas TLS (`tests/integration/test_tls.sh`). Falta prueba de CA vencida (RS-07) y fuzzing.

## Pendientes tecnicos conocidos de la etapa HTTP en claro

- ~~`getaddrinfo` en binario estatico~~ **Resuelto**: con glibc estatico la placa abortaba (`_dl_call_libc_early_init: Assertion sym != NULL`) al cargar NSS. El target `arm` ahora usa la toolchain musl de musl.cc (`arm-linux-musleabihf`, sha256 fijado en el Dockerfile), cuyo resolver no usa NSS y que ya trae `explicit_bzero`.
- `getaddrinfo` es bloqueante y detiene el reactor; pendiente cache/resolucion asincrona.
- Una solicitud por conexion: se envia `Connection: close` al servidor; no hay keep-alive ni pool todavia.
- No se eliminan los encabezados nombrados dentro de `Connection:` (solo la lista fija de hop-by-hop).
- Un solo worker; el modelo de un worker por nucleo queda para despues.
- `Upgrade` (WebSocket) se pasa como tunel; `Expect: 100-continue` pasa tal cual por el relay.

## DES-3.4: bitacora de bloqueos (SQLite)

- **Que se guarda** (`--block-log FILE`, tabla `blocks`): `ts` (epoch UTC), `host` (en minusculas, sin puerto, sin ruta ni query), `score` (NULL si no hubo: acelerador caido, timeout...), `umbral` vigente en ese momento y `reason` (`nsfw`, `accel_down`, `accel_timeout`, `queue_full`, `accel_error`...). Nunca bytes de imagen ni URLs (RS-05); `test_blocklog.sh` comprueba que ni la ruta ni la query llegan al archivo `.db`/`-wal`.
- **Solo bloqueos**: las imagenes permitidas no dejan fila, tampoco las no examinadas que la politica deja pasar.
- **Esquema en ingles** (`blocks`, `host`, `reason`...), no el de la guia (`bloqueos`, `dominio`, `motivo`): el codigo va en ingles y `fpgate-admin` leera por la API de control, no el `.db`.
- **Escritura**: hilo propio con cola de 256 filas; las filas de una rafaga comparten transaccion (espera de 250 ms). WAL + `synchronous=NORMAL`: un corte de energia puede perder los ultimos bloqueos, pero no corromper el archivo. Si la cola se llena se descarta la fila y se cuenta en `blocklog_stats().dropped`; el proxy nunca espera a la microSD.
- **SQLite** viene del amalgamation 3.50.4 (sha256 fijado en el Dockerfile), estatico en `/opt/sqlite/{native,arm}`, sin extensiones cargables.
- Pendiente: consulta paginada expuesta por la API de control (`blocklog_page` ya existe), y definir si hay rotacion/limite de tamano del archivo.

## DES-3.4: interfaz de administracion

- **Donde vive**: dentro del proxy (no en un proceso `fpgate-admin` aparte, como sugeria la guia), en el mismo reactor: la capa TLS comparte DRBG y cache de hojas, y un hilo aparte las pisaria. HTTPS en el puerto 8443 (`--admin-port`), con una hoja firmada por la CA de FPGate emitida para **la IP a la que entro el navegador** (`getsockname`), asi que instalar la CA una vez basta. Se entra por IP, no por nombre.
- **Paginas**: `web/login.html`, `dashboard.html`, `certificate.html` (plantillas de `Main/Dashboard/Certificado.html` con la API real) se compilan dentro del binario con `tools/embed_web.py`; un solo archivo que copiar a la placa. Se sirven con `Connection: close`, CSP, `X-Frame-Options: DENY` y `no-store`. Los textos de la bitacora se insertan con `textContent` (los dominios vienen de la red).
- **Login (RS-03)**: archivo `user:iteraciones:sal:hash` con PBKDF2-HMAC-SHA256 (20 000 iteraciones, ~0,1 s en el A9, bloquea el reactor ese rato), modo 600, creado con `--set-admin-password`; **no hay contrasena por defecto**. Sesiones en memoria (token aleatorio de 128 bits, cookie `HttpOnly; Secure; SameSite=Strict`, 30 min de inactividad, 8 h maximo, 8 a la vez). Cinco intentos fallidos seguidos bloquean todos los logins 30 s (protege contra fuerza bruta a costa de que quien ataque tambien bloquee a la persona administradora). Los POST rechazan un `Origin` de otro sitio.
- **API** (JSON, mensajes en espanol y codigo estable para RU-06): `POST /api/login|logout|threshold|substitute`, `GET /api/state|blocklog|substitute`. Sin sesion todo da 401 salvo `/`, `/certificado` y `/ca.crt`, que son publicos (solo la CA publica, nunca la clave).
- **`http://fpgate.setup`**: a traves del proxy (puerto 8080) sirve la pagina del certificado y `ca.crt`, sin login; no expone la API. Funciona aunque la interfaz de administracion este apagada.
- **Configuracion (TOML)**: `--config` guarda solo lo editable en caliente: `[clasificacion] umbral` y `[imagenes] sustitucion`. Lo que dice el archivo pisa a `--threshold`/`--substitute`. Al guardar se reescribe solo esa linea (los comentarios y las demas claves se conservan; un comentario en la misma linea del valor se pierde). `SIGHUP` vuelve a leerlo; si es invalido se conserva lo vigente y el log dice por que. Sin `--config` los cambios aplican pero no sobreviven a un reinicio (la pagina lo avisa).
- **Imagen de sustitucion**: JPG/PNG/WebP, 2 MB, se decodifica para validarla antes de aceptarla; se guarda como `substitute.{png,jpg,webp}` junto al TOML.
- **Estadisticas**: latencia = tiempo de inferencia que reporta el acelerador (min/prom/max); throughput = respuestas por segundo en el ultimo minuto. Estados: `normal`, `degradado`, `config_inicial` (sin acelerador).
- Pendiente: socket Unix de control para otros clientes (la guia lo preve; hoy la API es la HTTPS), vigencia de la CA en el dashboard con fecha, avisar en la UI cuando se vence la CA (hoy solo dice que HTTPS no se revisa), rotacion de la bitacora, HSTS.

- **Acceso por nombre (`https://fpgate.admin`)**: la IP de la placa cambia (DHCP), asi que la entrada principal no la usa. Un `CONNECT fpgate.admin:443` que llega al proxy no se intercepta ni se reenvia: se contesta `200` y el socket (duplicado, tras quitarlo del epoll) pasa a la interfaz, que habla TLS con una hoja para `fpgate.admin`. El nombre no cambia, asi que el certificado siempre vale. `http://fpgate.admin` redirige a https. Tambien se puede entrar directo a `:8443`: ese certificado lleva el nombre de la placa, `<nombre>.local`, `localhost` y la IP usada. Un `CONNECT` a la propia IP:8443 desde un navegador que usa el proxy se deja pasar como tunel (sin interceptar).

## Cierre de DES-3: fuzzing y CA vencida

- **Fuzzing** (`make fuzz-run SECS=N`, `tests/fuzz/`): libFuzzer con ASan + UBSan sobre seis blancos que reciben datos no confiables: framing HTTP (`http_stream`, con pausas `hs_stop/hs_rest` y cortes en cualquier byte), detector de imagen / Content-Type / Accept (`detect`), decodificadores JPEG/PNG/WebP con el remuestreo (`image`), archivo TOML con su reescritura (`toml`), archivo de credenciales (`creds`) y el parser de solicitudes del panel con su JSON y la subida de imagen (`admin`; se compila `admin.c` dentro del arnes, con la sesion siempre valida, para llegar a las rutas autenticadas). TLS queda fuera: es mbedTLS. Las semillas se generan al correr (`make_corpus.py`), no se versionan binarios.
- **Hallazgos**: ninguno en el codigo del proxy. Un fallo inicial era del arnes (el nombre de una cabecera plegada es NULL en picohttpparser). El ASan de clang 14 se cae al arrancar en algunas ejecuciones (WSL2 / Docker Desktop, entropia de 32 bits en la ASLR): `run.sh` las detecta (segfault con log vacio) y repite.
- **CA vencida (RS-07)** (`test_ca_expiry.sh`, usa `faketime` para fabricar una CA ya vencida o que vence en 22 s): vencida al arrancar o durante la ejecucion, el proxy deja de interceptar y HTTPS pasa como tunel ciego (el navegador ve el certificado real, no se revisan esas imagenes), el HTTP en claro se sigue inspeccionando, el panel HTTPS deja de responder y `http://fpgate.setup` sigue entregando el certificado. Cambios: la vigencia se revisa cada **10 s** (antes 60) y el log dice la fecha de vencimiento al arrancar. **Defecto corregido**: con una CA ya vencida al arrancar no se escribia ningun aviso (el mensaje solo salia al cambiar de estado, y se partia de "apagado").
- Limitacion: al vencer la CA, HTTPS queda sin revisar hasta que se renueve. Es la politica pedida por la guia (tunel ciego) y no la de bloquear todo HTTPS; si se prefiere "fallar cerrado", seria un cambio de una linea en `on_upstream_connect`, a costa de dejar sin HTTPS a los dispositivos de prueba.
- **Corrida de cierre** (180 s por blanco, 2026-11-08): http_stream 139 347 ejecuciones, detect 1 703 170, image 63 634, toml 12 546, creds 70 069, admin 129 472; sin fallos, fugas ni timeouts. Limite honesto: libjpeg/libpng/libwebp no estan compilados con instrumentacion, asi que ASan detecta ahi errores de memoria pero libFuzzer no se guia por su cobertura; el blanco `image` explora sobre todo nuestro codigo y el de remuestreo. Una corrida mas larga (horas) del `image` y de `http_stream` queda como mejora si se quiere mas confianza.

## RS-01 / RS-05: sin persistencia de imagenes (`test_no_persist.sh`)

- La prueba pasa por el proxy imagenes con una marca unica (PNG y JPEG con la marca en sus metadatos) y una con un color de pixel distintivo, por todos los caminos: permitida, bloqueada, JPEG troceado, WebP, corrupta, mas grande que `-L` (se transmite sin retener), con `Content-Encoding`, en HTTP en claro y en HTTPS interceptado.
- **Disco**: el proxy corre con su propio directorio de trabajo, temporal y de configuracion. Lo unico que escribe es la bitacora SQLite (y el TOML si se usa el panel); su directorio de trabajo y su temporal quedan vacios; ningun otro archivo de la maquina cambia mientras corre; ni la marca ni los pixeles decodificados (el tensor RGB que va al acelerador) aparecen en esos archivos ni en su log; la bitacora solo tiene `id, ts, host, score, threshold, reason`.
- **Memoria**: con el proceso aun vivo tras el trafico, se recorre toda su memoria legible (19 MB) y no queda ni la marca ni los pixeles (hay una cadena de control para comprobar que el escaneo lee de verdad). Esto cubre el `explicit_bzero` de RS-01: las copias de la imagen y del tensor se borran al liberarse. Se ejecuta con el binario nativo, no con ASan, porque ASan retiene la memoria liberada en cuarentena.
- Limites: no mira la memoria del nucleo (buffers de sockets) ni el intercambio (la placa no tiene swap configurado; verificarlo en la placa con `cat /proc/swaps`), y la imagen de sustitucion subida por la persona administradora si se guarda, a proposito.
