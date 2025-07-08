#!/bin/bash

DIR=$1
OUT=$2
if [ -z "$DIR" ] || [ -z "$OUT" ]; then
  echo "Usage: $0 <directory> <executable-name>";
  exit 1;
fi

CFLAGS="-Wall -Wextra -ggdb -Ofast `pkg-config --cflags raylib`"
LIBS="`pkg-config --libs raylib` -lm"

echo "Building $DIR -> $OUT"
set -xe;
clang $CFLAGS -o $OUT ./$DIR/*.c $LIBS -L./bin/
