#!/bin/bash


set -xe

CFLAGS="-Wall -Wextra -ggdb -Ofast `pkg-config --cflags raylib`"
LIBS="`pkg-config --libs raylib` -lm"

clang $CFLAGS -o ./net ./*.c $LIBS -L./bin/
