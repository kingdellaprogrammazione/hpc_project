#include "consts.h"
#include "structs.h"
#include <stdio.h>
#include "mpi_functions.h"

void produce_csv(FILE **matrix_file_output, char *filename, float *matrix_final, int matrix_side)
{

    if ((*matrix_file_output = fopen(filename, "w")) == NULL)
    {
        printf("Error while opening csv files.\n");
    }

    for (int row = 0; row < matrix_side; row++)
    {
        for (int col = 0; col < matrix_side; col++)
        {

            fprintf(*matrix_file_output, "%f", matrix_final[row * matrix_side + col]);
            if (col != matrix_side - 1)
            {
                fprintf(*matrix_file_output, ",");
            }
        }
        fprintf(*matrix_file_output, "\n");
    }
}

int get_csv_matrix_dimension(FILE *file) // return matrix dimensions if valid
{
    char *line = NULL; // Line buffer (will be allocated by getline)
    size_t len = 0;    // Size of the buffer (getline sets it)

    int counter_lines_A = 0;
    int counter_columns_A = -1;

    while (getline(&line, &len, file) != -1)
    {
        // let's check here that the matrices have all the same dimensions
        line[strcspn(line, "\r\n")] = 0;

        int new_counter_cols = 0;

        char *token;
        token = strtok(line, ",");
        while (token != NULL)
        {
            token = strtok(NULL, ",");
            new_counter_cols++;
        }

        if (counter_columns_A != -1)
        {
            if (counter_columns_A != new_counter_cols)
            {
                rewind(file); // Good practice to reset the file state before returning.
                free(line);
                return 0;
            }
        }
        counter_columns_A = new_counter_cols;
        counter_lines_A++;
        // perform the check
    }

    if (!(counter_columns_A == counter_lines_A))
    {
        rewind(file); // Good practice to reset the file state before returning.
        free(line);
        return 0;
    }

    rewind(file);
    free(line);
    return counter_columns_A;
}

void obtain_full_matrices(const char *filename_matrix_A, const char *filename_matrix_B, MPIContext *ctx, Block ***block_matrix_A, Block ***block_matrix_B)
// to be executed only from root process
{
    int status = STATUS_OK; // change status to context

    // printf("%d processes running.\n", world_size);

    FILE *file_matrix_A = NULL;
    FILE *file_matrix_B = NULL;
    // FILE *file_matrix_C = NULL;

    if ((file_matrix_A = fopen(filename_matrix_A, "r")) == NULL)
    {
        printf("Error while opening csv files.\n");
        status = STATUS_READING_ERROR;
    }
    if (status == STATUS_OK)
        if ((file_matrix_B = fopen(filename_matrix_B, "r")) == NULL)
        {
            printf("Error while opening csv files.\n");
            status = STATUS_READING_ERROR;
        }

    // if ((file_matrix_C = fopen("matrix_c.csv", "w")) == NULL)
    // {
    //     printf("Error while opening csv files.\n");
    //     status = STATUS_READING_ERROR;
    // }

    // now detect the dimension of each matrix, ensure they are both square and have the same side
    // *matrix_A_side = get_csv_matrix_dimension(file_matrix_A);

    int matrix_A_side = get_csv_matrix_dimension(file_matrix_A);
    ctx->matrix_side = matrix_A_side;

    fprintf(ctx->log_file, "Matrix A dims %d\n", matrix_A_side);

    int matrix_B_side = get_csv_matrix_dimension(file_matrix_B);
    if (status == STATUS_OK)
    {
        if ((matrix_A_side != matrix_B_side) || (matrix_A_side == 0))
        {
            printf("Error while parsing csv file.\n");
            status = STATUS_MATRIX_INVALID;
        }
    }

    float *matrix_A = NULL;
    float *matrix_B = NULL;

    // now actually read mmatrix A
    matrix_A = matrix_reader(file_matrix_A, matrix_A_side);

    // now actually read matrix B
    matrix_B = matrix_reader(file_matrix_B, matrix_B_side);

    if (file_matrix_A != NULL)
        fclose(file_matrix_A);
    if (file_matrix_B != NULL)
        fclose(file_matrix_B);

    // now parse the matrices

    // now divide the matrix in block and do the systolic array multiplications in blocks.
    int block_size = ctx->matrix_side / ctx->processor_grid_side;
    int block_reminder = ctx->matrix_side % ctx->processor_grid_side;

    fprintf(ctx->log_file, "block size %d\nblock reminder %d\n", block_size, block_reminder);

    // analyze matrix A (and so B) structure
    ctx->matrix_block_structure = calculate_blocks_sizes(ctx->processor_grid_side, ctx->matrix_side);

    // divide matrix A in blocks according to the processor grid.
    *block_matrix_A = matrix_parser_general(matrix_A, ctx->matrix_block_structure, ctx->processor_grid_side);

    // divide matrix B in blocks according to the processor grid.
    *block_matrix_B = matrix_parser_general(matrix_B, ctx->matrix_block_structure, ctx->processor_grid_side);
}
