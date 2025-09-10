#include "structs.h"
#include "consts.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Block *block_create(int num_rows, int num_cols, float *values)
{
    Block *temp_block;

    temp_block = (Block *)malloc(sizeof(Block));

    if (temp_block == NULL)
    {
        fprintf(stderr, "Failed to allocate memory for Block struct\n");
        return NULL; // Return NULL to signal failure
    }

    temp_block->dims[0] = num_rows;
    temp_block->dims[1] = num_cols;

    int total_dimension = num_rows * num_cols;

    if (total_dimension == 0)
    {
        temp_block->ptr_to_block_contents = NULL; // Handle zero-sized blocks
    }
    else
    {
        temp_block->ptr_to_block_contents = (float *)malloc(total_dimension * sizeof(float));
        // check for allocation failure
        if (temp_block->ptr_to_block_contents == NULL)
        {
            fprintf(stderr, "Failed to create Block\n");
            free(temp_block); // IMPORTANT: clean up partially allocated struct
            return NULL;      // Return NULL to signal failure
        }
        if (values != NULL)
        {
            memcpy(temp_block->ptr_to_block_contents, values, total_dimension * sizeof(float));
        }
    }
    return temp_block;
}

void block_destroy(Block *block_to_destroy)
{
    if (block_to_destroy != NULL)
    {
        free(block_to_destroy->ptr_to_block_contents); // Free the internal data first
        free(block_to_destroy);                        // Then free the struct itself
    }
}

void set_MPIContext(MPIContext *ctx)
{
    ctx->world_rank = 0;
    ctx->world_size = 0;
    ctx->processor_grid_side = 0;
    ctx->matrix_side = 0;

    ctx->cart_rank = -1;
    ctx->cart_size = 0;
    ctx->full_rank = -1;
    ctx->full_size = 0;

    // Communication topology
    ctx->sendto_vertical = -1;
    ctx->sendto_horizontal = -1;
    ctx->receivefrom_vertical = -1;
    ctx->receivefrom_horizontal = -1;

    ctx->local_block_rows = 0;
    ctx->local_block_cols = 0;

    ctx->coming_block_dims_vertical[0] = 0;
    ctx->coming_block_dims_vertical[1] = 0;

    ctx->going_block_dims_horizontal[0] = 0;
    ctx->going_block_dims_horizontal[1] = 0;

    ctx->coming_block_dims_horizontal[0] = 0;
    ctx->coming_block_dims_horizontal[1] = 0;

    ctx->going_block_dims_vertical[0] = 0;
    ctx->going_block_dims_vertical[1] = 0;

    ctx->cartesian_coords[0] = 0;
    ctx->cartesian_coords[1] = 0;

    ctx->status = STATUS_OK;

    ctx->full_group_comm = MPI_COMM_NULL;
    ctx->worker_comm = MPI_COMM_NULL;
    ctx->cart_comm = MPI_COMM_NULL;

    // Other
    ctx->matrix_block_structure = NULL;

    // Block data
    ctx->coming_block_vertical = NULL;
    ctx->going_block_horizontal = NULL;
    ctx->going_block_vertical = NULL;
    ctx->coming_block_horizontal = NULL;
    ctx->multi_result = NULL;
    ctx->local_block = NULL;

    snprintf(ctx->log_process_filepath, sizeof(ctx->log_process_filepath), "./data/logs/log_process_full_rank_");
    snprintf(ctx->extension, sizeof(ctx->extension), ".txt");

    ctx->log_file = NULL;
}

#include <stdio.h>
#include <mpi.h>

// Add STATUS_OK or your own status enum if needed
// #define STATUS_OK 0

void log_MPIContext(MPIContext *ctx)
{
    if (ctx->log_file != NULL)
    {
        fprintf(ctx->log_file, "=== MPIContext Dump ===\n");

        fprintf(ctx->log_file, "world_rank: %d\n", ctx->world_rank);
        fprintf(ctx->log_file, "world_size: %d\n", ctx->world_size);
        fprintf(ctx->log_file, "processor_grid_side: %d\n", ctx->processor_grid_side);
        fprintf(ctx->log_file, "matrix_side: %d\n", ctx->matrix_side);

        fprintf(ctx->log_file, "cart_rank: %d\n", ctx->cart_rank);
        fprintf(ctx->log_file, "cart_size: %d\n", ctx->cart_size);
        fprintf(ctx->log_file, "full_rank: %d\n", ctx->full_rank);
        fprintf(ctx->log_file, "full_size: %d\n", ctx->full_size);

        fprintf(ctx->log_file, "sendto_vertical: %d\n", ctx->sendto_vertical);
        fprintf(ctx->log_file, "sendto_horizontal: %d\n", ctx->sendto_horizontal);
        fprintf(ctx->log_file, "receivefrom_vertical: %d\n", ctx->receivefrom_vertical);
        fprintf(ctx->log_file, "receivefrom_horizontal: %d\n", ctx->receivefrom_horizontal);

        fprintf(ctx->log_file, "local_block_rows: %d\n", ctx->local_block_rows);
        fprintf(ctx->log_file, "local_block_cols: %d\n", ctx->local_block_cols);

        fprintf(ctx->log_file, "coming_block_dims_vertical: [%d, %d]\n",
                ctx->coming_block_dims_vertical[0],
                ctx->coming_block_dims_vertical[1]);
        fprintf(ctx->log_file, "coming_block_dims_horizontal: [%d, %d]\n",
                ctx->coming_block_dims_horizontal[0],
                ctx->coming_block_dims_horizontal[1]);

        fprintf(ctx->log_file, "going_block_dims_vertical: [%d, %d]\n",
                ctx->going_block_dims_vertical[0],
                ctx->going_block_dims_vertical[1]);
        fprintf(ctx->log_file, "going_block_dims_horizontal: [%d, %d]\n",
                ctx->going_block_dims_horizontal[0],
                ctx->going_block_dims_horizontal[1]);

        fprintf(ctx->log_file, "cartesian_coords: [%d, %d]\n",
                ctx->cartesian_coords[0],
                ctx->cartesian_coords[1]);

        fprintf(ctx->log_file, "status: %d\n", ctx->status);

        fprintf(ctx->log_file, "full_group_comm: %s\n", ctx->full_group_comm == MPI_COMM_NULL ? "MPI_COMM_NULL" : "Valid MPI_Comm");
        fprintf(ctx->log_file, "worker_comm: %s\n", ctx->worker_comm == MPI_COMM_NULL ? "MPI_COMM_NULL" : "Valid MPI_Comm");
        fprintf(ctx->log_file, "cart_comm: %s\n", ctx->cart_comm == MPI_COMM_NULL ? "MPI_COMM_NULL" : "Valid MPI_Comm");

        fprintf(ctx->log_file, "matrix_block_structure: %p\n", (void *)ctx->matrix_block_structure);

        fprintf(ctx->log_file, "coming_block_vertical: %p\n", (void *)ctx->coming_block_vertical);
        fprintf(ctx->log_file, "going_block_horizontal: %p\n", (void *)ctx->going_block_horizontal);
        fprintf(ctx->log_file, "going_block_vertical: %p\n", (void *)ctx->going_block_vertical);
        fprintf(ctx->log_file, "coming_block_horizontal: %p\n", (void *)ctx->coming_block_horizontal);
        fprintf(ctx->log_file, "multi_result: %p\n", (void *)ctx->multi_result);
        fprintf(ctx->log_file, "local_block: %p\n", (void *)ctx->local_block);

        fprintf(ctx->log_file, "log_process_filepath: %s\n", ctx->log_process_filepath);
        fprintf(ctx->log_file, "extension: %s\n", ctx->extension);

        fprintf(ctx->log_file, "log_file: %p\n", (void *)ctx->log_file);
        fprintf(ctx->log_file, "========================\n");
    }
}

void open_logfiles(MPIContext *ctx)
{ // to call after one has defined full_rank

    char number[5];
    snprintf(number, sizeof(number), "%d", ctx->full_rank);
    strcat(ctx->log_process_filepath, number);
    strcat(ctx->log_process_filepath, ctx->extension);

    if ((ctx->log_file = fopen(ctx->log_process_filepath, "w")) == NULL)
    {
        printf("error in opening log files\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// TODO close logfiles