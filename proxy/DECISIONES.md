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
