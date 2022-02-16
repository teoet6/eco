#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "pishtov.h"

// #define FIELD_W 240
// #define FIELD_H 135
#define FIELD_W 960
#define FIELD_H 540

#define INITIAL_CELLS_LEN 1000
#define SYNAPSES_LEN 50
#define MUTATION_CHANCE .00001f
#define MINIMUM_METABOLISM .02f

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
    new->prev = NULL;

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

void create_random_cell() {
    struct Cell *new = alloc_cell(&cells_head);

    int64_t tries = 0;
    do {
        new->x = rand64() % FIELD_W;
        new->y = rand64() % FIELD_H;
        if (++tries > 100) return;
    } while (field[new->x][new->y]);
    {
        int8_t dir = rand() % 4;
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

    field[new->x][new->y] = new;
}

void init() {
    uint32_t seed = get_timestamp();
    printf("    seed = 0x%08x;\n", seed);
    srand(seed);

    for (uint64_t i = 0; i < INITIAL_CELLS_LEN; ++i) {
        create_random_cell(&cells_head);
    }
}

float get_in_like(struct Cell *c, int8_t dx, int8_t dy) {
    struct Cell *other = field[mod(c->x + dx, FIELD_W)][mod(c->y + dy, FIELD_H)];
    if (!other) return 0.f;
    return other->color == c->color ? 1.f : -1.f;
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
        free_cell(&cells_head, field[c->x][c->y]);
        field[c->x][c->y] = c;
        return false;
    } else {
        field[c->x][c->y]->energy = energy_sum;
        free_cell(&cells_head, c);
        return true;
    }
}

void update_brain(struct Cell *c) {
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

    float new_neurons[NEURONS_LEN] = {};

    for (int32_t i = 0; i < SYNAPSES_LEN; ++i) {
        new_neurons[c->synapses[i].dst] += c->neurons[c->synapses[i].src] * c->synapses[i].weight;
    }

    memcpy(c->neurons, new_neurons, sizeof(c->neurons));
}

void kill_cell(struct Cell *c) {
    field[c->x][c->y] = NULL;
    free_cell(&cells_head, c);
}
void mutate(struct Cell *c, float mutation_chance) {
    if (frandf() < mutation_chance) {
        c->color = rand64() & 0xffffff;
        c->metabolism = frandf() + MINIMUM_METABOLISM;
    }

    for (uint64_t i = 0; i < SYNAPSES_LEN; ++i) {
        if (frandf() < mutation_chance) {
            c->color = rand64() & 0xffffff;
            c->synapses[i].src = rand64() % NEURONS_LEN;
            c->synapses[i].dst = rand64() % NEURONS_LEN;
            c->synapses[i].weight = frandf() * 2.f - 1.f;
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
    struct Cell *new = alloc_cell(&cells_head);
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

    new->energy *= .5f;
    c  ->energy *= .5f;

    place_on_field_or_die(new);
}

void act_based_on_brain(struct Cell *c) {
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

    update_brain(c);
    act_based_on_brain(c);

    struct Cell *next = c->next;
    if (place_on_field_or_die(c)) return next;
    return c->next;
}

void do_tick() {
    static struct Cell *cur;

    if (!cur) {
        create_random_cell();
        cur = cells_head;
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

    for (struct Cell *it = cells_head; it; it = it->next) {
        uint32_t color = (it->sleeping * 0x808080) | (!it->sleeping * it->color);
        fill_color(color);

        fill_rect(it->x, it->y, 1.f, 1.f);
    }
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
