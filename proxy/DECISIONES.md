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
| GIF, WebP, AVIF/HEIC, BMP | entregado, `unsupported_format` | `unsupported_format` |
| JPEG CMYK/YCCK o de 12 bits | entregado, `unsupported_format` | `unsupported_format` |
| Mas de 16 MP (`max_pixels`) | entregado, `too_many_pixels` | `too_many_pixels` |
| Mayor que `-L` (8 MiB) | se transmite sin retener | `skipped_oversize` |
| Presupuesto global de retencion (64 MiB) agotado | se transmite sin retener | `skipped_budget` |
| 206 Partial Content | se transmite sin analizar | `skipped_partial` |
| `Content-Encoding` distinto de identity | se transmite sin analizar | `skipped_encoded` |
| Content-Length + Transfer-Encoding, longitudes contradictorias, chunk mal formado | **se cierra la conexion** (un parser que discrepa del navegador permitiria colar imagenes) | - |

**Abierto**: ninguno de estos casos *bloquea* todavia; en DES-3.3 hay que decidir si WebP/GIF/AVIF y las imagenes danadas se bloquean (fail-closed, rompe muchos sitios porque WebP es muy comun) o se dejan pasar (hueco de contenido). Mi recomendacion: configurable, con "permitir" por defecto para formatos no soportados y "bloquear" para danadas. Decision de producto, no tecnica.

**Memoria y RS-01**: las imagenes retenidas viven en buffers que se copian y se borran con `explicit_bzero` al crecer y al liberar; lo mismo con filas y pixeles temporales en los decodificadores. Nada se escribe a disco. El redimensionado se hace fila a fila mientras se decodifica, asi que una foto de 24 MP no necesita su RGB completo (solo `alto x 448 x 3` bytes).

**Tiempos medidos en la placa** (2026-11-02, Cortex-A9 con NEON, binario estatico musl, salida 448 crop, `prep_tool`):

| Imagen | Decodificar + redimensionar | Con `-D` |
|---|---|---|
| JPEG 1280x575, 216 KB | 92 ms | - |
| PNG 800x600 RGBA, 219 KB | 118 ms | - |
| JPEG 3840x1725, 1.5 MB | 537 ms | 277 ms |

Por el proxy, JPEG 1280x575 por HTTPS interceptado: `prep=92.0ms` en el log, 200 en 1.06 s totales de punta a punta (incluye handshakes y la descarga desde internet). La resolucion y el codec dominan: extrapolando por pixeles, una foto de 16 MP (el tope) tardaria ~1.3 s exacta y ~0.7 s con `-D`.

**Limitaciones que quedan**
- La decodificacion corre **dentro del reactor** y bloquea el resto de conexiones mientras dura: 90-540 ms para imagenes tipicas y hasta ~1.3 s en el peor caso permitido, contra los 500 ms de RR-03 para el trafico sin imagenes. **Hay que moverla a un hilo de trabajo antes de DES-3.3** (el HPS tiene 2 nucleos, pero `fpgate-accel` tambien usa CPU del HPS para las operaciones por fila).
- Un JPEG progresivo grande necesita sus coeficientes en memoria (limite de libjpeg: 96 MB).
- Sin pipelining >32 solicitudes en vuelo (pasa a opaco sin analizar), sin HTTP/2 (ALPN lo impide).
- El presupuesto de retencion es global y fijo (64 MiB); la concurrencia maxima de imagenes en vuelo no esta limitada de otra forma (decision #8 sigue abierta en este punto).
- Fuzzing del detector, del framing y de los decodificadores: pendiente.

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
