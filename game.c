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

    struct Cell *next;
    struct Cell *prev;
};

struct Cell *field[FIELD_W][FIELD_H];

struct Cell *cells_head;

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

struct Cell *alloc_cell(struct Cell **head) {
    struct Cell *new = malloc(sizeof(*new));

    new->next = *head;
    if (*head) new->prev = (*head)->prev;
    else new->prev = NULL;

    if (new->prev) new->prev->next = new;
    if (new->next) new->next->prev = new;

    *head = new;

    return new;
}

void free_cell(struct Cell **head, struct Cell *c) {
    if (c->prev) c->prev->next = c->next;
    if (c->next) c->next->prev = c->prev;
    if (*head == c) *head = c->next;
    free(c);
}

void init() {
    srand(get_timestamp());
    // srand(1);

    for (int i = 0; i < INITIAL_CELLS_LEN; ++i) {
        struct Cell *new = alloc_cell(&cells_head);

        do {
            new->x = rand64() % FIELD_W;
            new->y = rand64() % FIELD_H;
        } while (field[new->x][new->y]);
        field[new->x][new->y] = new;

        new->color = rand64() & 0xffffff;
        new->metabolism = frandf();
        new->energy = 1.f;
        for (int64_t i = 0; i < SYNAPSES_LEN; ++i) {
            new->synapses[i].src = rand64() % NEURONS_LEN;
            new->synapses[i].dst = rand64() % NEURONS_LEN;
            new->synapses[i].weight = frandf() * 2.f - 1.f;
        }
    }
}

void update_brain(struct Cell *c) {
    c->neurons[IN_BIAS] = 1.f;
    c->neurons[IN_CELL_N] = field[c->x][mod(c->y - 1, FIELD_H)] ? 1 : -1;
    c->neurons[IN_CELL_E] = field[mod(c->x - 1, FIELD_W)][c->y] ? 1 : -1;
    c->neurons[IN_CELL_S] = field[c->x][mod(c->y + 1, FIELD_H)] ? 1 : -1;
    c->neurons[IN_CELL_W] = field[mod(c->x + 1, FIELD_W)][c->y] ? 1 : -1;
    c->neurons[IN_ENERGY] = c->energy * 2.f - 1.f;

    float new_neurons[NEURONS_LEN];
    for (int32_t i = 0; i < NEURONS_LEN; ++i) {
        new_neurons[i] = 0;
    }

    for (int32_t i = 0; i < SYNAPSES_LEN; ++i) {
        new_neurons[c->synapses[i].dst] += c->neurons[c->synapses[i].src] * c->synapses[i].weight;
    }

    memcpy(c->neurons, new_neurons, sizeof(c->neurons));
}

void kill_cell(struct Cell *c) {
    field[c->x][c->y] = NULL;
    free_cell(&cells_head, c);
}

void place_on_field(struct Cell *c) {
    if (field[c->x][c->y]) free_cell(&cells_head, field[c->x][c->y]);
    field[c->x][c->y] = c;
}

void do_mitose(struct Cell *c, int64_t x, int64_t y) {
    struct Cell *new = alloc_cell(&cells_head);
    {
        struct Cell *next = c->next;
        struct Cell *prev = c->prev;
        *new = *c;
        new->next = next;
        new->prev = prev;
    }

    new->x = x;
    new->y = y;

    place_on_field(new);

    new->energy *= .5f;
    c  ->energy *= .5f;
}

void act_based_on_brain(struct Cell *c) {
    int32_t max_neuron_id = OUT_MOVE_N;
    for (int32_t i = 0; i < NEURONS_LEN; ++i) {
        if (c->neurons[max_neuron_id] < c->neurons[i]) max_neuron_id = i;
    }

    switch (max_neuron_id) {
    case OUT_MOVE_N: c->y = mod(c->y - 1, FIELD_H); place_on_field(c); break;
    case OUT_MOVE_E: c->x = mod(c->x - 1, FIELD_W); place_on_field(c); break;
    case OUT_MOVE_S: c->y = mod(c->y + 1, FIELD_H); place_on_field(c); break;
    case OUT_MOVE_W: c->x = mod(c->x + 1, FIELD_W); place_on_field(c); break;

    case OUT_MITOSE_N: do_mitose(c, c->x, mod(c->y - 1, FIELD_H)); break;
    case OUT_MITOSE_E: do_mitose(c, mod(c->x - 1, FIELD_W), c->y); break;
    case OUT_MITOSE_S: do_mitose(c, c->x, mod(c->y + 1, FIELD_H)); break;
    case OUT_MITOSE_W: do_mitose(c, mod(c->x + 1, FIELD_W), c->y); break;

    case OUT_SLEEP: c->sleeping = true; break;
    }
}

struct Cell *update_cell(struct Cell *c) {
    if (c->sleeping) {
        c->energy += c->metabolism;
        if (c->energy >= 1.f) {
            c->energy = 1.f;
            c->sleeping = false;
        }
        return c->next;
    }


    c->energy -= c->metabolism;
    if (c->energy <= 0.f) {
        struct Cell *next = c->next;
        kill_cell(c);
        return next;
    }

    field[c->x][c->y] = NULL;

    update_brain(c);
    act_based_on_brain(c);

    return c->next;
}

void do_tick() {
    static struct Cell *cur;
    if (!cur) cur = cells_head;
    cur = update_cell(cur);
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

    for (struct Cell *it = cells_head; it; it = it->next) {
        if (it->sleeping)
            fill_color(0x808080);
        else              fill_color(it->color);
        fill_ellipse(it->x + .5f, it->y + .5f, .5f, .5f);
    }
    // printf("Nigger\n");
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
