#!/bin/sh

# gcc --std=c11 game.c -Wall -Werror -ldl -lX11 -lGL -lm -g -o game
gcc -fno-sanitize-recover=undefined  -fsanitize=address --std=c11 game.c -Wall -Werror -ldl -lX11 -lGL -lm -g -o game
