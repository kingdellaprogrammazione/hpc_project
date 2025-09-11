#include <stdio.h>
#include "mpi.h"

#ifndef STRUCTS_H
#define STRUCTS_H

// Host matrices with dimensions
typedef struct
{
    float *ptr_to_block_contents;
    int dims[2];
} Block;

// Creates a Block
Block *block_create(int num_rows, int num_cols, float *values);

// Destroys a Block
void block_destroy(Block *block_to_destroy);

// Holds the local variables (for each processor)
typedef struct
{
    int world_rank;
    int world_size;
    int processor_grid_side;
    int matrix_side;

    int cart_rank;
    int cart_size;
    int full_rank;
    int full_size;

    int cartesian_coords[2];

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

    MPI_Comm full_group_comm;
    MPI_Comm worker_comm;
    MPI_Comm cart_comm;

} MPIContext;

// Zero-initializes all the local fields
void set_MPIContext(MPIContext *ctx);

// Print to log files all the local field for inspection
void log_MPIContext(MPIContext *ctx);

#endif