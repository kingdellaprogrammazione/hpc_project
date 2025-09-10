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

    ctx->status = STATUS_OK;

    // Other
    ctx->matrix_block_structure = NULL;

    // Matrix pointers
    ctx->matrix_A_read = NULL;
    ctx->matrix_B_read = NULL;
    ctx->matrix_C_read = NULL;

    ctx->block_matrix_A = NULL;
    ctx->block_matrix_B = NULL;

    // Block data
    ctx->coming_block_vertical = NULL;
    ctx->going_block_horizontal = NULL;
    ctx->going_block_vertical = NULL;
    ctx->coming_block_horizontal = NULL;
    ctx->multi_result = NULL;
    ctx->local_block = NULL;

    snprintf(ctx->log_process_filepath, sizeof(ctx->log_process_filepath), "./data/logs/log_process_");
    snprintf(ctx->extension, sizeof(ctx->extension), ".txt");

    ctx->log_file = NULL;
}