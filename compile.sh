#!/bin/sh

# CFLAGS='-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer --std=c11 -Wall -Werror -ggdb'
# CFLAGS='-ffast-math --std=c11 -Wall -Werror -O2 -march=native -g'
CFLAGS='-ffast-math --std=c11 -Wall -Werror -O2 -march=native'

LFLAGS='-ldl -lX11 -lGL -lm'

gcc $CFLAGS game.c $LFLAGS -o game
