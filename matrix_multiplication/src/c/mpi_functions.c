#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "structs.h"

#include "consts.h"

#define MATRIX_INVALID_DIMENSIONS 0

int check_square(int squared)
{
    int square_root = (int)round(sqrt((double)squared));
    if (square_root * square_root == squared)
    {
        return square_root;
    }
    return 0;
}

int find_int_sqroot(int number)
{
    return (int)sqrt((double)number); // IEE 754 guarantee
}

// buf is a pointer to pointers that points to memory locations with the submatrices // assume perfect dimensions for now
float **matrix_parser(float *matrix, int block_dimension, int block_matrix_side, int real_matrix_dimension) // block dimension is the dimension of the single block; block_matrix_side is the number of blocks we have in a side
{
    void **ptr_matrix_of_submatrices;                                                                   // pointer to other pointers
    ptr_matrix_of_submatrices = (void *)malloc(block_matrix_side * block_matrix_side * sizeof(void *)); // allocate a long 1D array of pointers (row major)
    // Step 3: Check for allocation failure
    if (ptr_matrix_of_submatrices == NULL)
    {
        fprintf(stderr, "Failed to allocate memory.\n");
        return NULL; // Exit with an error
    }
    int block_row = 0;
    int block_col = 0;

    // assume we have a long 1D array with the true values of the matrix we read before.

    for (int i = 0; i < block_matrix_side * block_matrix_side; i++)
    { // start the cycle on the array of pointers to blocks
        block_row = i / block_matrix_side;
        block_col = i % block_matrix_side;

        // evaluate matrix correct position
        int matrix_start_row_position = block_row * block_dimension;
        int matrix_start_col_position = block_col * block_dimension;

        float *ptr_to_the_block;
        ptr_to_the_block = (float *)malloc(block_dimension * block_dimension * sizeof(float)); // this is the 1D array of float numbers
        if (ptr_to_the_block == NULL)
        {
            fprintf(stderr, "Failed to allocate memory.\n");
            return NULL; // Exit with an error
        }

        for (int j = 0; j < block_dimension * block_dimension; j++)
        {
            int matrix_row_position = matrix_start_row_position + j / block_dimension;
            int matrix_col_position = matrix_start_col_position + j % block_dimension;

            int matrix_in_1d_array_position = matrix_row_position * real_matrix_dimension + matrix_col_position;
            ptr_to_the_block[j] = matrix[matrix_in_1d_array_position];
        }

        ptr_matrix_of_submatrices[i] = ptr_to_the_block; // if i do not put i I am overwriting the address of the first pointer with the address of the block.
    }
    return (float **)ptr_matrix_of_submatrices;
}

int *calculate_blocks_sizes(int processor_grid_side, int real_matrix_side) // calculate the ideal dimensions of blocks the processors will parse (calculation is the same for both matrices + they are squared so we need only a dimension)
{
    int *row_array = NULL;                                        // pointer to other pointers
    row_array = (int *)malloc(processor_grid_side * sizeof(int)); // allocate a long 1D array of pointers (row major)
    // check for allocation failure
    if (row_array == NULL)
    {
        fprintf(stderr, "Failed to allocate memory.\n");
        return NULL; // Exit with an error
    }
    int division = real_matrix_side / processor_grid_side;
    int reminder = real_matrix_side % processor_grid_side;

    for (int i = 0; i < processor_grid_side; i++)
    {
        if (reminder != 0)
        {
            row_array[i] = division + 1;
            reminder--;
        }
        else
        {
            row_array[i] = division;
        }
    }

    return row_array;

    // this function is important since i need the array to give to the function that splits and handle the matrices, and to the one that do algebra
}

Block **matrix_parser_general(float *matrix, int *block_structure, int processor_grid_side) // general function to read a square matrix and decompose it in blocks to be assigned to grid
{
    int real_matrix_side = 0;

    for (int i = 0; i < processor_grid_side; i++) // calculate real matrix dimension
    {
        real_matrix_side += block_structure[i];
    }

    void **ptr_matrix_of_submatrices = NULL;                                                                // pointer to other pointers
    ptr_matrix_of_submatrices = (void *)malloc(processor_grid_side * processor_grid_side * sizeof(void *)); // allocate a long 1D array of pointers (row major)
    // check for allocation failure
    if (ptr_matrix_of_submatrices == NULL)
    {
        fprintf(stderr, "Failed to allocate memory.\n");
        return NULL; // Exit with an error
    }

    // assume we have a long 1D array with the true values of the matrix we read before.

    for (int i = 0; i < processor_grid_side * processor_grid_side; i++)
    { // start the cycle on the array of pointers to blocks
        int processor_grid_index_row = i / processor_grid_side;
        int processor_grid_index_column = i % processor_grid_side;

        int matrix_start_column_position = 0;
        int matrix_start_row_position = 0;

        int block_total_elements = block_structure[processor_grid_index_row] * block_structure[processor_grid_index_column];

        for (int j = 0; j < processor_grid_index_row; j++)
        {
            matrix_start_row_position += block_structure[j]; // TODO CHECK
        }
        for (int j = 0; j < processor_grid_index_column; j++)
        {
            matrix_start_column_position += block_structure[j];
        }

        float *temp_container = NULL;                                          // can't create it locally, risk of stack overflow
        temp_container = (void *)malloc(block_total_elements * sizeof(float)); // allocate a long 1D array of pointers (row major)
        // check for allocation failure
        if (temp_container == NULL)
        {
            fprintf(stderr, "Failed to allocate memory.\n");
            return NULL; // Exit with an error
        }

        for (int j = 0; j < block_total_elements; j++)
        {
            int matrix_row_position = matrix_start_row_position + j / block_structure[processor_grid_index_column];    // i'm dividing j by the number of columns TODO CHECK FOR BUG
            int matrix_col_position = matrix_start_column_position + j % block_structure[processor_grid_index_column]; // we need always the columns

            int matrix_in_1d_array_position = matrix_row_position * real_matrix_side + matrix_col_position;

            temp_container[j] = matrix[matrix_in_1d_array_position];
        }

        Block *new_block = NULL;
        new_block = block_create(block_structure[processor_grid_index_row], block_structure[processor_grid_index_column], temp_container);

        ptr_matrix_of_submatrices[i] = new_block; // if i do not put i I am overwriting the address of the first pointer with the address of the block.

        free(temp_container);
    }

    return (Block **)ptr_matrix_of_submatrices;
}

void destroy_matrix_of_blocks(Block **matrix_of_blocks, int processor_grid_side)
{
    for (int i = 0; i < processor_grid_side * processor_grid_side; i++)
    {
        block_destroy(matrix_of_blocks[i]);
    }
    free(matrix_of_blocks);
}

float *matrix_reader(FILE *file, int real_matrix_dimension)
{

    char line[MAX_LINE_SIZE];
    int i = 0;

    float *matrix;                                                                           // pointer to other pointers
    matrix = (float *)malloc(real_matrix_dimension * real_matrix_dimension * sizeof(float)); // allocate a long 1D array of pointers (row major)
    // Step 3: Check for allocation failure
    if (matrix == NULL)
    {
        fprintf(stderr, "Failed to allocate memory.\n");
        return NULL; // Exit with an error
    }

    while (fgets(line, sizeof(line), file))
    {
        // let's check here that the matrices have all the same dimensions
        line[strcspn(line, "\r\n")] = 0;

        char *token;
        token = strtok(line, ",");

        while (token != NULL)
        {
            float token_to_number = atof(token);
            matrix[i] = token_to_number;
            token = strtok(NULL, ",");
            i++;
        }
    }
    return matrix;
}

void show_original_matrix(float *matrix, int side)
{
    for (int i = 0; i < side; i++)
    {
        for (int j = 0; j < side; j++)
        {
            printf("%f ", matrix[i * side + j]);
        }
        printf("\n");
    }
}

void show_matrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", matrix[i * rows + j]);
        }
        printf("\n");
    }
}

void show_blocks(float **matrix, int row, int col, int block_dimension, int block_matrix_side)
{
    float *block = matrix[row * block_matrix_side + col];
    for (int i = 0; i < block_dimension; i++)
    {
        for (int j = 0; j < block_dimension; j++)
        {
            printf("%f ", block[i * block_dimension + j]);
        }
        printf("\n");
    }
}

void show_blocks_general(Block **matrix, int row, int col, int processor_grid_side) // check for errors in row and col access
{
    if (row >= processor_grid_side || col >= processor_grid_side)
    {
        fprintf(stderr, "Tentative access to over-the-bounds indexes.\n");
        return;
    }

    Block *block = matrix[row * processor_grid_side + col];

    if (block == NULL)
    {
        printf("Block at (%d, %d) is NULL.\n", row, col);
        return;
    }

    // Use block->num_rows and block->num_cols for clarity
    for (int i = 0; i < block->dims[0]; i++)
    {
        for (int j = 0; j < block->dims[1]; j++)
        {
            printf("%f ", block->ptr_to_block_contents[i * block->dims[1] + j]); // we need to multiply i that is the index of the row, with the number of elements of the row, so the number of columns
        }
        printf("\n");
    }
}

float *create_null_matrix(int side)
{
    float *matrix = NULL;
    matrix = (float *)malloc(sizeof(float) * side * side);
    if (matrix == NULL)
    {
        fprintf(stderr, "Failed to allocate memory.\n");
        return NULL; // Exit with an error
    }
    return matrix;
}

float *create_zero_matrix(int total_elements)
{
    float *matrix = NULL;
    matrix = (float *)malloc(sizeof(float) * total_elements);
    if (matrix == NULL)
    {
        fprintf(stderr, "Failed to allocate memory.\n");
        return NULL; // Exit with an error
    }
    for (int i = 0; i < total_elements; i++)
    {
        matrix[i] = 0.0;
    }
    return matrix;
}

// matrix C needs to be already allocated
int matrix_multi(float *matrix_A, float *matrix_B, float *matrix_C, int num_rows_A, int common, int num_cols_B) // obviously the num_rows and num_cols refer to the first matrix, the second one needs to satisfy the exchanged dimensions
{
    for (int row_c = 0; row_c < num_rows_A; row_c++)
    {
        for (int col_c = 0; col_c < num_cols_B; col_c++) // TODO only num rows should be relevant right??    NO CHECK PERFORMED ON DIMENSIONS  COMMON is col of A or rows of B
        {
            float result = 0;

            for (int counter = 0; counter < common; counter++)
            {
                result = result + matrix_A[row_c * common + counter] * matrix_B[counter * num_cols_B + col_c]; // TODO is this right??????
            }

            matrix_C[row_c * num_cols_B + col_c] = result;
        }
    }
    return 0;
}

int matrix_add(float *matrix_A, float *matrix_B, float *matrix_C, int num_rows, int num_cols) // for addition it is sufficient to evaluate block dimension based on one of the two starting matrices.
{
    // The total number of elements is num_rows * num_cols
    int total_elements = num_rows * num_cols;

    // A simple loop over the 1D array is the most efficient way to do this
    for (int i = 0; i < total_elements; i++)
    {
        matrix_C[i] = matrix_A[i] + matrix_B[i];
    }

    return 0;
}

void from_blocks_to_matrix(float *matrix, float **target, int *matrix_block_structure, int grid_processor_side, int matrix_side)
{
    // strategia: per ogni elemento della mnatrice nuova contare in che blocco si trova e la sua posizione all'interno del blocco. una volta che abbiamo queste usiamo gli indici nella matrice vecchia per accedere a quello.
    int block_rows = 0;
    int block_cols = 0;

    float *new_matrix = NULL;

    new_matrix = (float *)malloc(matrix_side * matrix_side * sizeof(float));

    int *cumulated_block_structure;

    cumulated_block_structure = (int *)malloc(grid_processor_side * sizeof(int));

    cumulated_block_structure[0] = 0;
    for (int i = 1; i < grid_processor_side; i++)
    {
        cumulated_block_structure[i] = matrix_block_structure[i - 1] + cumulated_block_structure[i - 1];
        printf("cum_block_struct %d\n", cumulated_block_structure[i]);
    }

    // devo sfalsare i blocchi che sono contigui con in mezzo offset di lunghezza pari al resto della linea libero

    for (int row = 0; row < matrix_side; row++)
    {
        block_rows = 0; // row in which the block is situated
        for (int i = 0; i < grid_processor_side; i++)
        {
            if (row >= cumulated_block_structure[i])
            {
                block_rows = i;
            }
        }

        int position_inside_block_row = row - cumulated_block_structure[block_rows]; // caluclate the relative position inside the block

        for (int col = 0; col < matrix_side; col++)
        {
            // belonging block

            block_cols = 0;
            for (int i = 0; i < grid_processor_side; i++)
            {
                if (col >= cumulated_block_structure[i])
                {
                    block_cols = i;
                }
            }

            int position_inside_block_col = col - cumulated_block_structure[block_cols];
            printf("ROW %d, COL %d, block_row  %d and block_col %d position block row %d, position block col %d \n", row, col, block_rows, block_cols, position_inside_block_row, position_inside_block_col);

            // now access the old matrix

            // convert the blocks into positions and then add also the thing.

            int block_number = block_rows * grid_processor_side + block_cols;
            int start_position_in_old_matrix = 0;

            for (int i = 0; i < block_number; i++)
            {
                start_position_in_old_matrix += matrix_block_structure[i / grid_processor_side] * matrix_block_structure[i % grid_processor_side];
            }

            int end_position_in_old_matrix = start_position_in_old_matrix + position_inside_block_row * matrix_block_structure[block_cols] + position_inside_block_col;
            printf("Block number %d , start_position_in_old_matrix %d end position in old matrix %d\n", block_number, start_position_in_old_matrix, end_position_in_old_matrix);
            new_matrix[row * matrix_side + col] = (float)matrix[end_position_in_old_matrix];
        }

        *target = new_matrix;
    }
}
