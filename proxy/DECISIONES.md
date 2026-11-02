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
