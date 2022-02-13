#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "pishtov.h"
#include "arr.h"

#define FIELD_W 100
#define FIELD_H 100
#define INITIAL_CELLS_LEN 1000
#define SYNAPSES_LEN 20

float ticks_per_second = INITIAL_CELLS_LEN * 10.f;
float seconds_since_last_tick = 0;

enum Neuron_Id {
    IN_BIAS,
    IN_CELL_N,
    IN_CELL_E,
    IN_CELL_S,
    IN_CELL_W,
    IN_ENERGY,

    INTERNAL_A,
    INTERNAL_B,

    OUT_MOVE_N,
    OUT_MOVE_E,
    OUT_MOVE_S,
    OUT_MOVE_W,
    OUT_MITOSE_N,
    OUT_MITOSE_E,
    OUT_MITOSE_S,
    OUT_MITOSE_W,
    OUT_SLEEP,

    NEURONS_LEN,
};

struct Cell {
    int64_t x;
    int64_t y;
    uint32_t color;
    float energy;
    float metabolism;
    bool sleeping;

    float neurons[NEURONS_LEN];
    struct {
        int64_t src;
        int64_t dst;
        float weight;
    } synapses[SYNAPSES_LEN];
};

int64_t field[FIELD_W][FIELD_H];

struct Cell *cells;

int64_t mod(const int64_t x, const int64_t m) {
    return ((x % m) + m) % m;
}

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

    // memset(field, -1, FIELD_W * FIELD_H * sizeof(field[0][0]));
    for (int64_t x = 0; x < FIELD_W; ++x) {
        for (int64_t y = 0; y < FIELD_W; ++y) {
            field[x][y] = -1;
        }
    }

    for (struct Cell *it = cells; it < arr_end(cells); ++it) {
        do {
            it->x = rand64() % FIELD_W;
            it->y = rand64() % FIELD_H;
        } while (field[it->x][it->y] != -1);
        field[it->x][it->y] = it - cells;

        it->color = rand64() & 0xffffff;
        it->metabolism = frandf();
        it->energy = 1.f;
        for (int64_t i = 0; i < SYNAPSES_LEN; ++i) {
            it->synapses[i].src = rand64() % NEURONS_LEN;
            it->synapses[i].dst = rand64() % NEURONS_LEN;
            it->synapses[i].weight = frandf() * 2.f - 1.f;
        }
    }
}

void update_brain(size_t idx) {
    cells[idx].neurons[IN_BIAS] = 1.f;
    cells[idx].neurons[IN_CELL_N] = field[cells[idx].x][mod(cells[idx].y - 1, FIELD_H)] ? 1 : -1;
    cells[idx].neurons[IN_CELL_E] = field[mod(cells[idx].x - 1, FIELD_W)][cells[idx].y] ? 1 : -1;
    cells[idx].neurons[IN_CELL_S] = field[cells[idx].x][mod(cells[idx].y + 1, FIELD_H)] ? 1 : -1;
    cells[idx].neurons[IN_CELL_W] = field[mod(cells[idx].x + 1, FIELD_W)][cells[idx].y] ? 1 : -1;
    cells[idx].neurons[IN_ENERGY] = cells[idx].energy * 2.f - 1.f;

    float new_neurons[NEURONS_LEN];
    for (int32_t i = 0; i < NEURONS_LEN; ++i) {
        new_neurons[i] = 0;
    }

    for (int32_t i = 0; i < SYNAPSES_LEN; ++i) {
        new_neurons[cells[idx].synapses[i].dst] += cells[idx].neurons[cells[idx].synapses[i].src] * cells[idx].synapses[i].weight;
    }

    memcpy(cells[idx].neurons, new_neurons, sizeof(cells[idx].neurons));
}

void kill_cell(int64_t *idx) {
    cells[field[cells[*idx].x][cells[*idx].y]] = cells[arr_len(cells) - 1];
    arr_pop(&cells);
}

void place_on_field(int64_t *idx) {
    cells[field[cells[*idx].x][cells[*idx].y]] = cells[arr_len(cells) - 1];
    if (*idx == arr_len(cells) - 1) *idx = arr_len(cells) - 1;
    arr_pop(&cells);
}

void do_mitose(int64_t *idx, int64_t x, int64_t y) {
    arr_push(&cells, cells[*idx]);

    int64_t new = arr_len(cells) - 1;

    cells[new].x = x;
    cells[new].y = y;

    place_on_field(&new);

    cells[*idx].energy *= .5f;
    cells[new].energy *= .5f;
}

void act_based_on_brain(int64_t *idx) {
    int32_t max_neuron_id = OUT_MOVE_N;
    for (int32_t i = 0; i < NEURONS_LEN; ++i) {
        if (cells[*idx].neurons[max_neuron_id] < cells[*idx].neurons[i]) max_neuron_id = i;
    }

    switch (max_neuron_id) {
    case OUT_MOVE_N: cells[*idx].y = mod(cells[*idx].y - 1, FIELD_H); break;
    case OUT_MOVE_E: cells[*idx].x = mod(cells[*idx].x - 1, FIELD_W); break;
    case OUT_MOVE_S: cells[*idx].y = mod(cells[*idx].y + 1, FIELD_H); break;
    case OUT_MOVE_W: cells[*idx].x = mod(cells[*idx].x + 1, FIELD_W); break;

    case OUT_MITOSE_N: do_mitose(idx, cells[*idx].x, mod(cells[*idx].y - 1, FIELD_H)); break;
    case OUT_MITOSE_E: do_mitose(idx, mod(cells[*idx].x - 1, FIELD_W), cells[*idx].y); break;
    case OUT_MITOSE_S: do_mitose(idx, cells[*idx].x, mod(cells[*idx].y + 1, FIELD_H)); break;
    case OUT_MITOSE_W: do_mitose(idx, mod(cells[*idx].x + 1, FIELD_W), cells[*idx].y); break;

    case OUT_SLEEP: cells[*idx].sleeping = true; return;
    }
}

void update_cell(int64_t *idx) {
    if (cells[*idx].sleeping) {
        cells[*idx].energy += cells[*idx].metabolism;
        if (cells[*idx].energy >= 1.f) {
            cells[*idx].energy = 1.f;
            cells[*idx].sleeping = false;
        }
        return;
    }

    cells[*idx].energy -= cells[*idx].metabolism;
    if (cells[*idx].energy < 0.f) {
        kill_cell(idx);
        return;
    }

    field[cells[*idx].x][cells[*idx].y] = -1;

    update_brain(*idx);

    act_based_on_brain(idx);

    place_on_field(idx);
}

void do_tick() {
    static int64_t idx;

    update_cell(&idx);

    idx = (idx + 1) % arr_len(cells);
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
        if (it->sleeping) fill_color(0x808080);
        else              fill_color(it->color);
        fill_ellipse(it->x + .5f, it->y + .5f, .5f, .5f);
    }
}

void keydown(int key) {
    printf("%d\n", key);
    switch (key) {
    case 38:
        ticks_per_second *= 2.f;
        printf("%.0f tps\n", ticks_per_second);
        break;
    case 40:
        ticks_per_second *= .5f;
        printf("%.0f tps\n", ticks_per_second);
        break;
    }
}

void keyup(int key) {}

void mousedown(int button) {}

void mouseup(int button) {}
