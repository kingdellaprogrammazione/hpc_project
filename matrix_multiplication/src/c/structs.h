#ifndef STRUCTS_H
#define STRUCTS_H

typedef struct
{
    float *ptr_to_block_contents;
    int dims[2];
} Block;

Block *block_create(int num_rows, int num_cols, float *values);
void block_destroy(Block *block_to_destroy);

#endif