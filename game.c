#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "pishtov.h"
#include "arr.h"

#define MOD(X, M) (((X) + (M)) % (M))

#define FIELD_W 100
#define FIELD_H 100
#define INITIAL_CELLS_LEN 1000
#define SYNAPSES_LEN 10

float ticks_per_second = 1000;
float seconds_since_last_tick = 0;

enum Neuron_Id {
    INPUT_BIAS,
    INPUT_CELL_N,
    INPUT_CELL_E,
    INPUT_CELL_S,
    INPUT_CELL_W,

    OUTPUT_MOVE_NS,
    OUTPUT_MOVE_EW,

    NEURONS_LEN,
};

struct Brain {
    float neurons[NEURONS_LEN];
    struct {
        int64_t src;
        int64_t dst;
        float weight;
    } synapses[SYNAPSES_LEN];
};

struct Cell {
    int64_t x;
    int64_t y;
    uint32_t color;
    bool dead;
    struct Brain brain;
};

struct Cell *field[FIELD_W][FIELD_H];

struct Cell *cells;

uint64_t rand64() {
    return
        (uint64_t) rand()       ^
        (uint64_t) rand() << 15 ^
        (uint64_t) rand() << 30 ^
        (uint64_t) rand() << 45 ^
        (uint64_t) rand() << 60;
}

float frandf() {
    return (float) rand() / (float) RAND_MAX;
}

uint64_t get_timestamp() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

void init() {
    srand(get_timestamp());

    cells = arr_create(struct Cell);

    arr_resize(&cells, INITIAL_CELLS_LEN);

    for (struct Cell *it = cells; it < arr_end(cells); ++it) {
        do {
            it->x = rand64() % FIELD_W;
            it->y = rand64() % FIELD_H;
        } while (field[it->x][it->y]);
        field[it->x][it->y] = it;

        it->color = rand64() & 0xffffff;
        it->dead = false;
        for (int64_t i = 0; i < SYNAPSES_LEN; ++i) {
            it->brain.synapses[i].src = rand64() % SYNAPSES_LEN;
            it->brain.synapses[i].dst = rand64() % SYNAPSES_LEN;
            it->brain.synapses[i].weight = sqrt(frandf());
        }
    }
}

void update_brain(struct Brain *b);

void update_cell(struct Cell *c) {
    if (c->dead) return;

    field[c->x][c->y] = NULL;

    int64_t dx = rand() % 3 - 1;
    int64_t dy = rand() % 3 - 1;

    // update_brain(struct Brain *b);
    c->x += dx;
    c->y += dy;

    c->x = MOD(c->x, FIELD_W);
    c->y = MOD(c->y, FIELD_H);

    if (field[c->x][c->y]) field[c->x][c->y]->dead = true;

    field[c->x][c->y] = c;
}

void do_cleanup() {
    int64_t shift = 0;
    for (int64_t i = 0; i < arr_len(cells); ++i) {
        if (cells[i].dead) {
            ++shift;
            continue;
        }

        cells[i-shift] = cells[i];
        field[cells[i-shift].x][cells[i-shift].y] = &cells[i-shift];
    }
    arr_resize(&cells, arr_len(cells)-shift);
}

void do_tick() {
    static int64_t idx;

    while (cells[idx].dead) {
        ++idx;
        if (idx == arr_len(cells)) {
            do_cleanup();
            idx = 0;
        }
    }

    update_cell(&cells[idx]);

    ++idx;
    if (idx == arr_len(cells)) {
        do_cleanup();
        idx = 0;
    }
}

void update() {
    static uint64_t prev_ts;
    if (!prev_ts) prev_ts = get_timestamp();
    float dt;
    {
        uint64_t cur_ts = get_timestamp();
        dt = (float)(cur_ts - prev_ts) / 1000000000;
        prev_ts = cur_ts;
    }

    seconds_since_last_tick += dt;

    while (seconds_since_last_tick > 1/ticks_per_second) {
        do_tick();
        seconds_since_last_tick -= 1/ticks_per_second;
    }
}

void draw() {
    scale(fmin(window_w, window_h) / FIELD_W, fmin(window_w, window_h) / FIELD_H);

    for (struct Cell *it = cells; it < arr_end(cells); ++it) {
        if (it->dead) continue;
        fill_color(it->color);
        fill_ellipse(it->x + .5f, it->y + .5f, .5f, .5f);
    }
}

void keydown(int key) {}

void keyup(int key) {}

void mousedown(int button) {}

void mouseup(int button) {}
