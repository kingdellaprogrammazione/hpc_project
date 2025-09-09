#include "structs.h"
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