#!/bin/sh
# Builds zlib, libpng, libjpeg-turbo and libwebp as static libraries for the native
# host (/opt/deps/native) and for the HPS (/opt/deps/arm, musl, NEON).
# Source URLs and hashes come from the environment (see the Dockerfile).
set -e

fetch() { # url sha256 dest
    curl -fsSL "$1" -o "$3"
    echo "$2  $3" | sha256sum -c -
}

W=/tmp/deps-src
mkdir -p $W
fetch "$ZLIB_URL"   "$ZLIB_SHA256"   $W/zlib.tar.gz
fetch "$LIBPNG_URL" "$LIBPNG_SHA256" $W/libpng.tar.xz
fetch "$LIBJPEG_URL" "$LIBJPEG_SHA256" $W/libjpeg.tar.gz
fetch "$LIBWEBP_URL" "$LIBWEBP_SHA256" $W/libwebp.tar.gz
mkdir $W/zlib $W/libpng $W/libjpeg $W/libwebp
tar -xzf $W/zlib.tar.gz    -C $W/zlib    --strip-components=1
tar -xJf $W/libpng.tar.xz  -C $W/libpng  --strip-components=1
tar -xzf $W/libjpeg.tar.gz -C $W/libjpeg --strip-components=1
tar -xzf $W/libwebp.tar.gz -C $W/libwebp --strip-components=1

for T in native arm; do
    P=/opt/deps/$T
    if [ "$T" = arm ]; then
        TRIPLE=arm-linux-musleabihf
        export CC=$TRIPLE-gcc AR=$TRIPLE-ar RANLIB=$TRIPLE-ranlib
        HOSTOPT="--host=$TRIPLE"
        JPEG_TOOLCHAIN="-DCMAKE_TOOLCHAIN_FILE=$W/arm.cmake"
        cat > $W/arm.cmake <<EOF
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_C_COMPILER $TRIPLE-gcc)
set(CMAKE_C_FLAGS_INIT "-march=armv7-a -mfpu=neon -mfloat-abi=hard -fPIC")
EOF
    else
        export CC=gcc AR=ar RANLIB=ranlib
        HOSTOPT=""
        JPEG_TOOLCHAIN=""
    fi
    mkdir -p $P

    # zlib
    (cd $W/zlib && make distclean >/dev/null 2>&1 || true
     CFLAGS="-O2 -fPIC" ./configure --static --prefix=$P >/dev/null
     make -j"$(nproc)" >/dev/null && make install >/dev/null)

    # libpng
    (cd $W/libpng && make distclean >/dev/null 2>&1 || true
     CFLAGS="-O2 -fPIC" CPPFLAGS="-I$P/include" LDFLAGS="-L$P/lib" \
         ./configure $HOSTOPT --prefix=$P --disable-shared --enable-static >/dev/null
     make -j"$(nproc)" >/dev/null && make install >/dev/null)

    # libjpeg-turbo
    rm -rf $W/jpeg-build-$T && mkdir $W/jpeg-build-$T
    (cd $W/jpeg-build-$T && cmake $W/libjpeg $JPEG_TOOLCHAIN \
         -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$P -DCMAKE_INSTALL_LIBDIR=lib \
         -DENABLE_SHARED=0 -DENABLE_STATIC=1 -DWITH_TURBOJPEG=0 -DWITH_JAVA=0 \
         -DREQUIRE_SIMD=1 -DCMAKE_POSITION_INDEPENDENT_CODE=1 >/dev/null
     make -j"$(nproc)" >/dev/null && make install >/dev/null)
    # libwebp (decoder only is used; the tools and the encoder front-ends are not built)
    rm -rf $W/webp-build-$T && mkdir $W/webp-build-$T
    (cd $W/webp-build-$T && cmake $W/libwebp $JPEG_TOOLCHAIN \
         -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$P -DCMAKE_INSTALL_LIBDIR=lib \
         -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=1 \
         -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF \
         -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF \
         -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_EXTRAS=OFF >/dev/null
     make -j"$(nproc)" >/dev/null && make install >/dev/null)
    unset CC AR RANLIB
done

rm -rf $W
ls /opt/deps/native/lib /opt/deps/arm/lib
