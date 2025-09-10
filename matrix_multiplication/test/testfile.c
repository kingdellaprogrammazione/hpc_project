#include "../external/unity/unity.h"
#include "../src/c/structs.h" // Your actual structs header
#include "../src/c/mpi_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#define MATRIX_INVALID_DIMENSIONS 0
// --- setUp and tearDown ---

void setUp(void)
{
    // Set up code, executed before each test
}

void tearDown(void)
{
    // Clean up code, executed after each test
}

// --- Test Cases for functions in structs.c ---

void test_block_create_and_destroy(void)
{
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Block *b = block_create(2, 3, values);

    TEST_ASSERT_NOT_NULL(b);
    TEST_ASSERT_EQUAL_INT(2, b->dims[0]); // Check rows
    TEST_ASSERT_EQUAL_INT(3, b->dims[1]); // Check cols
    TEST_ASSERT_NOT_NULL(b->ptr_to_block_contents);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(values, b->ptr_to_block_contents, 6);

    // Test destroy: This is implicitly tested by running with memory checkers like Valgrind.
    // We can't programmatically check if memory is freed, but we call it to ensure it doesn't crash.
    block_destroy(b);
}

void test_block_create_null_values(void)
{
    Block *b = block_create(4, 3, NULL);
    TEST_ASSERT_NOT_NULL(b);
    TEST_ASSERT_EQUAL_INT(4, b->dims[0]);
    TEST_ASSERT_EQUAL_INT(3, b->dims[1]);
    TEST_ASSERT_NOT_NULL(b->ptr_to_block_contents); // Memory should still be allocated
    block_destroy(b);
}

// --- Test Cases for functions in your main C file ---

void test_get_csv_matrix_dimension(void)
{
    FILE *fp = tmpfile();
    TEST_ASSERT_NOT_NULL(fp);

    fprintf(fp, "1,2,3\n4,5,6\n7,8,9\n");
    rewind(fp);
    TEST_ASSERT_EQUAL_INT(3, get_csv_matrix_dimension(fp));

    fseek(fp, 0, SEEK_SET);
    ftruncate(fileno(fp), 0);
    fprintf(fp, "1,2\n3,4\n");
    rewind(fp);
    TEST_ASSERT_EQUAL_INT(2, get_csv_matrix_dimension(fp));

    fseek(fp, 0, SEEK_SET);
    ftruncate(fileno(fp), 0);
    fprintf(fp, "1,2,3\n4,5\n");
    rewind(fp);
    TEST_ASSERT_EQUAL_INT(MATRIX_INVALID_DIMENSIONS, get_csv_matrix_dimension(fp));

    fclose(fp);
}

void test_check_square(void)
{
    TEST_ASSERT_EQUAL_INT(5, check_square(25));
    TEST_ASSERT_EQUAL_INT(0, check_square(26));
    TEST_ASSERT_EQUAL_INT(1, check_square(1));
}

void test_find_int_sqroot(void)
{
    TEST_ASSERT_EQUAL_INT(5, find_int_sqroot(25));
    TEST_ASSERT_EQUAL_INT(5, find_int_sqroot(26));        // Checks for truncation
    TEST_ASSERT_EQUAL_INT(5, find_int_sqroot(25.000001)); // Checks for truncation
}

// void test_matrix_parser(void)
// {
//     float matrix[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
//     float **submatrices = matrix_parser(matrix, 2, 2, 4);
//     TEST_ASSERT_NOT_NULL(submatrices);
//
//     float block0[] = {1, 2, 5, 6};
//     float block1[] = {3, 4, 7, 8};
//     float block2[] = {9, 10, 13, 14};
//     float block3[] = {11, 12, 15, 16};
//
//     TEST_ASSERT_EQUAL_FLOAT_ARRAY(block0, submatrices[0], 4);
//     TEST_ASSERT_EQUAL_FLOAT_ARRAY(block1, submatrices[1], 4);
//     TEST_ASSERT_EQUAL_FLOAT_ARRAY(block2, submatrices[2], 4);
//     TEST_ASSERT_EQUAL_FLOAT_ARRAY(block3, submatrices[3], 4);
//
//     for (int i = 0; i < 4; i++)
//         free(submatrices[i]);
//     free(submatrices);
// }

void test_calculate_blocks_sizes(void)
{
    int *sizes1 = calculate_blocks_sizes(2, 4);
    int expected1[] = {2, 2};
    TEST_ASSERT_EQUAL_INT_ARRAY(expected1, sizes1, 2);
    free(sizes1);

    int *sizes2 = calculate_blocks_sizes(3, 8);
    // Logic from your code seems to distribute the remainder as +1 from the start
    int expected2[] = {3, 3, 2};
    TEST_ASSERT_EQUAL_INT_ARRAY(expected2, sizes2, 3);
    free(sizes2);
}

void test_matrix_parser_general_and_destroy(void)
{
    float matrix[] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25};
    int block_structure[] = {3, 2}; // rows for each processor
    int processor_grid_side = 2;

    Block **matrix_of_blocks = matrix_parser_general(matrix, block_structure, processor_grid_side);
    TEST_ASSERT_NOT_NULL(matrix_of_blocks);

    // Expected contents of the blocks
    float block00_exp[] = {1, 2, 3, 6, 7, 8, 11, 12, 13};
    float block01_exp[] = {4, 5, 9, 10, 14, 15};
    float block10_exp[] = {16, 17, 18, 21, 22, 23};
    float block11_exp[] = {19, 20, 24, 25};

    // Check block (0,0)
    TEST_ASSERT_EQUAL_INT(3, matrix_of_blocks[0]->dims[0]);
    TEST_ASSERT_EQUAL_INT(3, matrix_of_blocks[0]->dims[1]);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(block00_exp, matrix_of_blocks[0]->ptr_to_block_contents, 9);

    // Check block (0,1)
    TEST_ASSERT_EQUAL_INT(3, matrix_of_blocks[1]->dims[0]);
    TEST_ASSERT_EQUAL_INT(2, matrix_of_blocks[1]->dims[1]);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(block01_exp, matrix_of_blocks[1]->ptr_to_block_contents, 6);

    // Check block (1,0)
    TEST_ASSERT_EQUAL_INT(2, matrix_of_blocks[2]->dims[0]);
    TEST_ASSERT_EQUAL_INT(3, matrix_of_blocks[2]->dims[1]);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(block10_exp, matrix_of_blocks[2]->ptr_to_block_contents, 6);

    // Check block (1,1)
    TEST_ASSERT_EQUAL_INT(2, matrix_of_blocks[3]->dims[0]);
    TEST_ASSERT_EQUAL_INT(2, matrix_of_blocks[3]->dims[1]);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(block11_exp, matrix_of_blocks[3]->ptr_to_block_contents, 4);

    // Test the destruction function
    destroy_matrix_of_blocks(matrix_of_blocks, processor_grid_side);
}

void test_matrix_reader(void)
{
    FILE *fp = tmpfile();
    TEST_ASSERT_NOT_NULL(fp);
    fprintf(fp, "1.1,2.2\n3.3,4.4\n");
    rewind(fp);

    float *matrix = matrix_reader(fp, 2);
    float expected[] = {1.1, 2.2, 3.3, 4.4};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, matrix, 4);

    free(matrix);
    fclose(fp);
}

void test_create_null_matrix(void)
{
    float *matrix = create_null_matrix(3);
    TEST_ASSERT_NOT_NULL(matrix);
    free(matrix);
}

void test_create_zero_matrix(void)
{
    float *matrix = create_zero_matrix(9);
    TEST_ASSERT_NOT_NULL(matrix);
    float expected[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, matrix, 9);
    free(matrix);
}

void test_matrix_add(void)
{
    float A[] = {1, 2, 3, 4};
    float B[] = {5, 6, 7, 8};
    float C[4];
    float expected[] = {6, 8, 10, 12};

    matrix_add(A, B, C, 2, 2);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, C, 4);
}

void test_matrix_multi(void)
{
    float A[] = {1, 2, 3,
                 4, 5, 6};
    float B[] = {4, 5, 6, 7,
                 6, 5, 4, 8,
                 4, 6, 5, 9};
    float C[8];
    float expected[] = {28, 33, 29, 50,
                        70, 81, 74, 122};

    matrix_multi(A, B, C, 2, 3, 4);
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, C, 8);
}

void test_from_blocks_to_matrix(void)
{
    float A[] = {4, 5, 6, 13, 22,
                 7, 8, 9, 14, 23,
                 10, 11, 12, 15, 24,
                 16, 17, 18, 19, 25,
                 30, 31, 32, 34, 35};
    int matrix_block_structure[2] = {3, 2};
    int matrix_side = 5;
    int processor_side = 2;

    float *target = NULL;

    from_blocks_to_matrix(A, &target, matrix_block_structure, processor_side, matrix_side);

    float expected[] = {4, 5, 6, 23, 10,
                        13, 22, 7, 11, 12,
                        8, 9, 14, 15, 24,
                        16, 17, 18, 31, 32,
                        19, 25, 30, 34, 35};
    TEST_ASSERT_EQUAL_FLOAT_ARRAY(expected, target, 16);
}

// --- Test Runner ---

int main(void)
{
    UNITY_BEGIN();
    RUN_TEST(test_block_create_and_destroy);
    RUN_TEST(test_block_create_null_values);
    RUN_TEST(test_get_csv_matrix_dimension);
    RUN_TEST(test_check_square);
    RUN_TEST(test_find_int_sqroot);
    // RUN_TEST(test_matrix_parser);
    RUN_TEST(test_create_zero_matrix);
    RUN_TEST(test_calculate_blocks_sizes);
    RUN_TEST(test_matrix_parser_general_and_destroy);
    RUN_TEST(test_matrix_reader);
    RUN_TEST(test_create_null_matrix);
    RUN_TEST(test_matrix_multi);
    RUN_TEST(test_matrix_add);
    RUN_TEST(test_from_blocks_to_matrix);
    return UNITY_END();
}