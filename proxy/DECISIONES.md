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

## #1 Biblioteca TLS — ABIERTA

- OpenSSL 3.x o mbedTLS compilada estatica para ARM. Decidir tras prototipo en la placa (memoria, tiempo de handshake, generacion de certificados al vuelo).

## Pendientes tecnicos conocidos de la etapa HTTP en claro

- ~~`getaddrinfo` en binario estatico~~ **Resuelto**: con glibc estatico la placa abortaba (`_dl_call_libc_early_init: Assertion sym != NULL`) al cargar NSS. El target `arm` ahora usa la toolchain musl de musl.cc (`arm-linux-musleabihf`, sha256 fijado en el Dockerfile), cuyo resolver no usa NSS y que ya trae `explicit_bzero`.
- `getaddrinfo` es bloqueante y detiene el reactor; pendiente cache/resolucion asincrona.
- Una solicitud por conexion: se envia `Connection: close` al servidor; no hay keep-alive ni pool todavia.
- No se eliminan los encabezados nombrados dentro de `Connection:` (solo la lista fija de hop-by-hop).
- Un solo worker; el modelo de un worker por nucleo queda para despues.
- `Upgrade` (WebSocket) se pasa como tunel; `Expect: 100-continue` pasa tal cual por el relay.
