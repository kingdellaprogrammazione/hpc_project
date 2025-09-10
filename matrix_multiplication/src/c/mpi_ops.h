#include "mpi.h"
#include "structs.h"

#ifndef MPI_OPS_H
#define MPI_OPS_H

void organize_processors(MPIContext *ctx);
void obtain_rank_conversion(MPIContext *ctx, int **full_comm_ranks, int **rank_lookup_table_cartesian);
void printonrank(char *string, int rank, MPI_Comm comm);

void cartesian_explorer(MPIContext *ctx);

#endif