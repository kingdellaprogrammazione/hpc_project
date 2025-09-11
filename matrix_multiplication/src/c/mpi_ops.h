#include "mpi.h"
#include "structs.h"

#ifndef MPI_OPS_H
#define MPI_OPS_H

void organize_processors(MPIContext *ctx);
void obtain_rank_conversion(MPIContext *ctx, int **full_comm_ranks, int **rank_lookup_table_cartesian);
void printonrank(char *string, int rank, MPI_Comm comm);

void cartesian_explorer(MPIContext *ctx);

int scatter_block_dims(MPIContext *ctx, Block **block_matrix_A);

void send_dimensions_matrix_left(MPIContext *ctx, int *rank_lookup_table, Block **block_matrix, int start_index, int end_index, int clock);
void send_dimensions_matrix_top(MPIContext *ctx, int *rank_lookup_table, Block **block_matrix, int start_index, int end_index, int clock);
void receive_exchange_dims_grid(MPIContext *ctx, int clock);

void send_blocks_matrix_left(MPIContext *ctx, int *rank_lookup_table, Block **block_matrix, int start_index, int end_index, int clock);
void send_blocks_matrix_top(MPIContext *ctx, int *rank_lookup_table, Block **block_matrix, int start_index, int end_index, int clock);
void receive_exchange_blocks_grid(MPIContext *ctx, int clock);

void run_local_calculation(MPIContext *ctx, int clock);

void prepare_next_clock(MPIContext *ctx);

void main_loop(MPIContext *ctx, int *rank_lookup_table, Block **block_matrix_A, Block **block_matrix_B);

#endif