#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "structs.h"
#include "consts.h"
#include "mpi_functions.h"
#include <mpi.h>

void obtain_full_matrices(const char *filename_matrix_A, const char *filename_matrix_B, float **matrix_A, float **matrix_B, int *matrix_A_side)
// to be executed only from root process
{
    int status = STATUS_OK;

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
    *matrix_A_side = get_csv_matrix_dimension(file_matrix_A);

    printf("Matrix A dims %d\n", *matrix_A_side);

    int matrix_B_side = get_csv_matrix_dimension(file_matrix_B);
    if (status == STATUS_OK)
    {
        if ((*matrix_A_side != matrix_B_side) || (*matrix_A_side == 0))
        {
            printf("Error while parsing csv file.\n");
            status = STATUS_MATRIX_INVALID;
        }
    }

    // now actually read mmatrix A
    *matrix_A = matrix_reader(file_matrix_A, *matrix_A_side);

    // now actually read matrix B
    *matrix_B = matrix_reader(file_matrix_B, matrix_B_side);

    if (file_matrix_A != NULL)
        fclose(file_matrix_A);
    if (file_matrix_B != NULL)
        fclose(file_matrix_B);
    // if (file_matrix_C != NULL)
    //     fclose(file_matrix_C);
}
