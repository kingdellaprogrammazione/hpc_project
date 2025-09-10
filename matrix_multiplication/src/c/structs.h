#include <stdio.h>

#ifndef STRUCTS_H
#define STRUCTS_H

typedef struct
{
    float *ptr_to_block_contents;
    int dims[2];
} Block;

Block *block_create(int num_rows, int num_cols, float *values);
void block_destroy(Block *block_to_destroy);

// struct for holding the indexes

typedef struct
{
    int world_rank;
    int world_size;
    int processor_grid_side;
    int matrix_side;

    // Matrix pointers
    float *matrix_A_read;
    float *matrix_B_read;
    float *matrix_C_read;

    Block **block_matrix_A;
    Block **block_matrix_B;

    // Communication topology
    int sendto_vertical;
    int sendto_horizontal;
    int receivefrom_vertical;
    int receivefrom_horizontal;

    // Block data
    float *coming_block_vertical;
    float *going_block_horizontal;
    float *going_block_vertical;
    float *coming_block_horizontal;
    float *multi_result;
    float *local_block;

    int local_block_rows;
    int local_block_cols;

    int coming_block_dims_vertical[2];
    int coming_block_dims_horizontal[2];
    int going_block_dims_vertical[2];
    int going_block_dims_horizontal[2];

    // File paths
    char log_process_filepath[64];
    char extension[8];

    // Other
    int *matrix_block_structure;
    int status;

    FILE *log_file;
} MPIContext;

void set_MPIContext(MPIContext *ctx);

#endif