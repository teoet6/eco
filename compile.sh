#!/bin/sh

time gcc --std=c11 game.c -ldl -lX11 -lGL -lm -O2 -g -o game
