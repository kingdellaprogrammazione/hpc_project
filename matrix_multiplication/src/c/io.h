#include <stdio.h>

#ifndef IO_H
#define IO_H

// return matrix side if valid, otherwise throws an error
int get_csv_matrix_dimension(FILE *file);

// Generates the target csv file from the matrix array
int produce_csv(FILE **matrix_file_output, char *filename, float *matrix_final, int matrix_side);

// Parses the input matrices
int obtain_full_matrices(const char *filename_matrix_A, const char *filename_matrix_B, MPIContext *ctx, Block ***block_matrix_A, Block ***block_matrix_B);

//  Opens the logfile with live logging (if live = 1)
int open_logfiles(MPIContext *ctx, int live);

#endif