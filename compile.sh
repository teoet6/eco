#!/bin/sh

gcc --std=c11 game.c -Wall -Werror -ldl -lX11 -lGL -lm -O2 -g -o game
