#include <stdio.h>

#ifndef IO_H
#define IO_H

int get_csv_matrix_dimension(FILE *file); // return matrix dimensions if valid

void produce_csv(FILE **matrix_file_output, char *filename, float *matrix_final, int matrix_side);

void obtain_full_matrices(const char *filename_matrix_A, const char *filename_matrix_B, MPIContext *ctx, Block ***block_matrix_A, Block ***block_matrix_B);

#endif