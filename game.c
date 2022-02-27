#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "pishtov.h"

#define PI 3.141592653589793238
#define E  2.718281828459045235

// #define FIELD_W 240
// #define FIELD_H 135
#define FIELD_W 960
#define FIELD_H 540

#define INITIAL_CELLS_LEN 1000
#define SYNAPSES_LEN 30
#define MUTATION_CHANCE .0001f
#define MINIMUM_METABOLISM 0.f
#define ENERGY_MULTIPLIED_AFER_MITOSIS .5f

float ticks_per_second = 1024000.f;
float seconds_since_last_tick = 0;

enum Neuron_Id {
    IN_BIAS,
    IN_LIKE_U,
    IN_LIKE_R,
    IN_LIKE_D,
    IN_LIKE_L,
    IN_EATABLE_U,
    IN_EATABLE_R,
    IN_EATABLE_D,
    IN_EATABLE_L,
    IN_NORTH_U,
    IN_NORTH_R,
    IN_NORTH_D,
    IN_NORTH_L,
    IN_ENERGY,

    INTERNAL_A,
    INTERNAL_B,
    INTERNAL_C,
    INTERNAL_D,
    INTERNAL_E,

    OUT_MOVE_U,
    OUT_MOVE_R,
    OUT_MOVE_D,
    OUT_MOVE_L,
    OUT_MITOSE_U,
    OUT_MITOSE_R,
    OUT_MITOSE_D,
    OUT_MITOSE_L,
    OUT_SLEEP,

    NEURONS_LEN,
};

enum Combining_Function_Id {
    COMB_SIGMOID,
    COMB_COS,
    COMB_LEN,
};

struct Cell {
    int64_t x;
    int64_t y;
    int8_t dir_x;
    int8_t dir_y;

    uint32_t color;
    float energy;
    float metabolism;
    bool sleeping;

    float neurons[NEURONS_LEN];
    enum Combining_Function_Id neuron_combs[NEURONS_LEN];
    struct {
        int64_t src;
        int64_t dst;
        float weight;
    } synapses[SYNAPSES_LEN];

    struct Cell *next;
    struct Cell *prev;
};

struct Cell_Arena {
    int64_t cap; // constant
    int64_t len;
    struct Cell *data;
    struct Cell **free;

    struct Cell *head;
};

struct Cell *field[FIELD_W][FIELD_H];

struct Cell_Arena cell_arena;

int64_t mod(const int64_t x, const int64_t m) {
    return ((x % m) + m) % m;
}

static unsigned long xorshf_x=123456789, xorshf_y=362436069, xorshf_z=521288629;

// xorshf96
// period 2^96-1
uint64_t rand64() {
    uint64_t t;
    xorshf_x ^= xorshf_x << 16;
    xorshf_x ^= xorshf_x >> 5;
    xorshf_x ^= xorshf_x << 1;

    t = xorshf_x;
    xorshf_x = xorshf_y;
    xorshf_y = xorshf_z;
    xorshf_z = t ^ xorshf_x ^ xorshf_y;

    return xorshf_z;
}

void srand64(uint64_t seed) {
    seed &= 0x0fffff;
    printf("    seed = 0x%lx;\n", seed);
    for (uint64_t i = 0; i < seed; ++i) rand64();
}

float frandf() {
    return (rand64() & 0xffffff) / 16777216.f;
}

uint64_t get_timestamp() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

void init_cell_arena(struct Cell_Arena *ca, int64_t cap) {
    ca->cap = cap;
    ca->len = 0;
    ca->head = NULL;
    ca->data = malloc((sizeof(*ca->data) + sizeof(*ca->free)) * ca->cap);
    ca->free = (void*)ca->data + sizeof(*ca->data) * ca->cap;

    for (int64_t i = 0; i < ca->cap; ++i) {
        ca->free[i] = ca->data + i;
    }
}

void deinit_cell_arena(struct Cell_Arena *ca) {
    free(ca->data);
}

struct Cell *alloc_cell(struct Cell_Arena *ca) {
    struct Cell *new = ca->free[ca->len++];

    new->next = ca->head;
    new->prev = NULL;

    if (new->next) new->next->prev = new;

    ca->head = new;

    return new;
}

void free_cell(struct Cell_Arena *ca, struct Cell *c) {
    if (c->prev) c->prev->next = c->next;
    if (c->next) c->next->prev = c->prev;
    if (ca->head == c) ca->head = c->next;

    ca->free[--ca->len] = c;
}

float comb_sigmoid(float x) {
    const float y = 1.f / (1.f + powf(E, 4.f * x));
    return -2.f * y + 1.f;
}

float comb_cos(float x) {
    if (x > 1.f || x < -1.f) return 1.f;
    return -cosf(PI * x);
}

void create_random_cell() {
    struct Cell *new = alloc_cell(&cell_arena);

    int64_t tries = 0;
    do {
        new->x = rand64() % FIELD_W;
        new->y = rand64() % FIELD_H;
        if (++tries > 100) return;
    } while (field[new->x][new->y]);
    {
        int8_t dir = rand64() % 4;
        new->dir_x = ( dir & 1) * -(dir >> 1);
        new->dir_y = (~dir & 1) * -(dir >> 1);
    }
    new->color = rand64() & 0xffffff;
    new->metabolism = frandf() + MINIMUM_METABOLISM;
    new->energy = 1.f;
    new->sleeping = false;
    for (int64_t i = 0; i < SYNAPSES_LEN; ++i) {
        new->synapses[i].src = rand64() % NEURONS_LEN;
        new->synapses[i].dst = rand64() % NEURONS_LEN;
        new->synapses[i].weight = frandf() * 2.f - 1.f;
    }
    for (int64_t i = 0; i < NEURONS_LEN; ++i) {
        new->neuron_combs[i] = rand64() % COMB_LEN;
    }

    field[new->x][new->y] = new;
}

void init() {
    srand64(get_timestamp());

    // Add ten cells of breathing space
    init_cell_arena(&cell_arena, FIELD_W * FIELD_H + 10);

    for (uint64_t i = 0; i < INITIAL_CELLS_LEN; ++i) {
        create_random_cell(&cell_arena);
    }
}

float get_in_like(struct Cell *c, int8_t dx, int8_t dy) {
    struct Cell *other = field[mod(c->x + dx, FIELD_W)][mod(c->y + dy, FIELD_H)];
    if (!other) return 0.f;
    uint32_t diff = other->color ^ c->color;
    diff = (diff & 0xff) | (diff >> 8 & 0xff) | (diff >> 16 & 0xff);
    return diff / 128.f - 1.f;
}

float get_in_eatable(struct Cell *c, int8_t dx, int8_t dy) {
    struct Cell *other = field[mod(c->x + dx, FIELD_W)][mod(c->y + dy, FIELD_H)];
    if (!other) return 0.f;
    return other->sleeping || c->energy > other->energy ? 1.f : -1.f;
}

// True if dead
bool place_on_field_or_die(struct Cell *c) {
    float eatable = get_in_eatable(c, 0, 0);

    if (eatable == 0.f) {
        field[c->x][c->y] = c;
        return false;
    }

    float energy_sum = fmin(1.f, c->energy + field[c->x][c->y]->energy);

    if (eatable == 1.f) {
        c->energy = energy_sum;
        free_cell(&cell_arena, field[c->x][c->y]);
        field[c->x][c->y] = c;
        return false;
    } else {
        field[c->x][c->y]->energy = energy_sum;
        free_cell(&cell_arena, c);
        return true;
    }
}

void set_brain_inputs(struct Cell *c) {
    c->neurons[IN_BIAS] = 1.f;

    c->neurons[IN_LIKE_U] = get_in_like(c,  c->dir_x,  c->dir_y);
    c->neurons[IN_LIKE_L] = get_in_like(c, -c->dir_y,  c->dir_x);
    c->neurons[IN_LIKE_D] = get_in_like(c, -c->dir_x, -c->dir_y);
    c->neurons[IN_LIKE_R] = get_in_like(c,  c->dir_x, -c->dir_x);

    c->neurons[IN_EATABLE_U] = get_in_eatable(c,  c->dir_x,  c->dir_y);
    c->neurons[IN_EATABLE_L] = get_in_eatable(c, -c->dir_y,  c->dir_x);
    c->neurons[IN_EATABLE_D] = get_in_eatable(c, -c->dir_x, -c->dir_y);
    c->neurons[IN_EATABLE_R] = get_in_eatable(c,  c->dir_x, -c->dir_x);

    c->neurons[IN_NORTH_U] = c->dir_y == -1 ? 1.f : -1.f;
    c->neurons[IN_NORTH_L] = c->dir_x == -1 ? 1.f : -1.f;
    c->neurons[IN_NORTH_D] = c->dir_y ==  1 ? 1.f : -1.f;
    c->neurons[IN_NORTH_R] = c->dir_x ==  1 ? 1.f : -1.f;

    c->neurons[IN_ENERGY] = c->energy * 2.f - 1.f;
}

void update_brain(struct Cell *c) {
    float new_neurons[NEURONS_LEN] = {};

    for (int32_t i = 0; i < SYNAPSES_LEN; ++i) {
        new_neurons[c->synapses[i].dst] += c->neurons[c->synapses[i].src] * c->synapses[i].weight;
    }

    for (int32_t i = 0; i < NEURONS_LEN; ++i) {
        switch (c->neuron_combs[i]) {
        case COMB_SIGMOID: c->neurons[i] = comb_sigmoid(new_neurons[i]); break;
        case COMB_COS:     c->neurons[i] = comb_cos    (new_neurons[i]); break;
        default: assert(false);
        }
    }
}

void kill_cell(struct Cell *c) {
    field[c->x][c->y] = NULL;
    free_cell(&cell_arena, c);
}

uint32_t similar_color(uint32_t col) {
    uint32_t r = (rand64() & 0xf0) | 0x10;
    uint32_t g = (rand64() & 0xf0) | 0x10;
    uint32_t b = (rand64() & 0xf0) | 0x10;

    r = (r & -r) - 1;
    g = (g & -g) - 1;
    b = (b & -b) - 1;

    return col ^ (r << 16 | g << 8 | b);
}

void mutate(struct Cell *c, float mutation_chance) {
    if (frandf() < mutation_chance) {
        c->metabolism = frandf() + MINIMUM_METABOLISM;
        c->color = similar_color(c->color);
    }

    for (uint64_t i = 0; i < SYNAPSES_LEN; ++i) {
        if (frandf() < mutation_chance) {
            c->synapses[i].src = rand64() % NEURONS_LEN;
            c->synapses[i].dst = rand64() % NEURONS_LEN;
            c->synapses[i].weight = frandf() * 2.f - 1.f;
            c->color = similar_color(c->color);
        }
    }

    for (uint64_t i = 0; i < NEURONS_LEN; ++i) {
        if (frandf() < mutation_chance) {
            c->neuron_combs[i] = rand64() % COMB_LEN;
            c->color = similar_color(c->color);
        }
    }
}

void do_move(struct Cell *c, int8_t dx, int8_t dy) {
    c->x = mod(c->x + dx, FIELD_W);
    c->y = mod(c->y + dy, FIELD_H);

    c->dir_x = dx;
    c->dir_y = dy;
}

void do_mitose(struct Cell *c, int8_t dx, int8_t dy) {
    struct Cell *new = alloc_cell(&cell_arena);
    {
        struct Cell *next = new->next;
        struct Cell *prev = new->prev;
        *new = *c;
        new->next = next;
        new->prev = prev;
    }

    mutate(new, MUTATION_CHANCE);

    do_move(new, dx, dy);
    c->dir_x = -dx;
    c->dir_y = -dy;

    new->energy *= ENERGY_MULTIPLIED_AFER_MITOSIS;
    c  ->energy *= ENERGY_MULTIPLIED_AFER_MITOSIS;

    place_on_field_or_die(new);
}

void act_based_on_brain_outputs(struct Cell *c) {
    int32_t max_neuron_id = OUT_MOVE_U;
    for (int32_t i = max_neuron_id; i < NEURONS_LEN; ++i) {
        if (c->neurons[max_neuron_id] < c->neurons[i]) max_neuron_id = i;
    }

    switch (max_neuron_id) {
    case OUT_MOVE_U: do_move(c,  c->dir_x,  c->dir_y); break;
    case OUT_MOVE_L: do_move(c, -c->dir_y,  c->dir_x); break;
    case OUT_MOVE_D: do_move(c, -c->dir_x, -c->dir_y); break;
    case OUT_MOVE_R: do_move(c,  c->dir_y, -c->dir_x); break;

    case OUT_MITOSE_U: do_mitose(c,  c->dir_x,  c->dir_y); break;
    case OUT_MITOSE_L: do_mitose(c, -c->dir_y,  c->dir_x); break;
    case OUT_MITOSE_D: do_mitose(c, -c->dir_x, -c->dir_y); break;
    case OUT_MITOSE_R: do_mitose(c,  c->dir_y, -c->dir_x); break;

    case OUT_SLEEP: c->sleeping = true; break;
    }
}

struct Cell *update_cell(struct Cell *c) {
    if (c->sleeping) {
        c->energy += c->metabolism;
        if (c->energy >= 1.f) {
            c->energy = 1.f;
            c->sleeping = false;
        } else {
            return c->next;
        }
    }


    c->energy -= c->metabolism;
    if (c->energy <= 0.f) {
        struct Cell *next = c->next;
        kill_cell(c);
        return next;
    }

    field[c->x][c->y] = NULL;

    set_brain_inputs(c);
    update_brain(c);
    act_based_on_brain_outputs(c);

    struct Cell *next = c->next;
    if (place_on_field_or_die(c)) return next;
    return c->next;
}

void do_tick() {
    static struct Cell *cur;

    if (!cur) {
        create_random_cell();
        cur = cell_arena.head;
    }
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
    {
        float min_scale = fminf(window_w / FIELD_W, window_h / FIELD_H);
        scale(min_scale, min_scale);
    }

    uint8_t buf[FIELD_W * FIELD_H * 4];
    memset(buf, 0xff, FIELD_W * FIELD_H * 4);

    for (struct Cell *it = cell_arena.head; it; it = it->next) {
        uint32_t color = it->sleeping ? 0x808080 : it->color;

        buf[4 * (it->y * FIELD_W + it->x) + 0] = color >> 16 & 0xff;
        buf[4 * (it->y * FIELD_W + it->x) + 1] = color >>  8 & 0xff;
        buf[4 * (it->y * FIELD_W + it->x) + 2] = color       & 0xff;
    }

    draw_image_buffer(buf, FIELD_W, FIELD_H, 0, 0, FIELD_W, FIELD_H);
}

void keydown(int key) {
    // printf("%d\n", key);
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
