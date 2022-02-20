#!/bin/sh

# CFLAGS='-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer --std=c11 -Wall -Werror -ggdb'
# CFLAGS='--std=c11 -Wall -Werror -O2 -ggdb'
CFLAGS='--std=c11 -Wall -Werror -O2'

LFLAGS='-ldl -lX11 -lGL -lm'

gcc $CFLAGS game.c $LFLAGS -o game
