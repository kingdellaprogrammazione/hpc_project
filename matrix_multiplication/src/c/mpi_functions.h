
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "structs.h"

#ifndef MYFUNCTIONS_H
#define MYFUNCTIONS_H

int get_csv_matrix_dimension(FILE *file);

int check_square(int squared);

int find_int_sqroot(int number);

float **matrix_parser(float *matrix, int subblock_dimension, int submatrices_side, int real_matrix_dimension);

float *matrix_reader(FILE *file, int real_matrix_dimension);

void show_original_matrix(float *matrix, int side);

void show_blocks(float **matrix, int row, int col, int sub_sides, int block_side);

float *create_null_matrix(int side);

float *create_zero_matrix(int total_elements);

int *calculate_blocks_sizes(int processor_grid_side, int real_matrix_side);

// matrix C needs to be already allocated
int matrix_multi(float *matrix_A, float *matrix_B, float *matrix_C, int num_rows_A, int common, int num_cols_B);

int matrix_add(float *matrix_A, float *matrix_B, float *matrix_C, int num_rows, int num_cols);

void show_blocks_general(Block **matrix, int row, int col, int processor_grid_side);

Block **matrix_parser_general(float *matrix, int *block_structure, int processor_grid_side); // general function to read a square matrix and decompose it in blocks to be assigned to grid

void destroy_matrix_of_blocks(Block **matrix_of_blocks, int processor_grid_side);

void show_matrix(float *matrix, int rows, int cols);

void from_blocks_to_matrix(float *matrix, float **target, int *matrix_block_structure, int grid_processor_side, int matrix_side);

void produce_csv(FILE **matrix_file_output, char *filename, float *matrix_final, int matrix_side);

#endif