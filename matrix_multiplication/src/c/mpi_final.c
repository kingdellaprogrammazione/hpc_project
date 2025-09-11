#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi_functions.h"
#include "structs.h"
#include <unistd.h>
#include "consts.h"
#include "io.h"
#include <assert.h>
#include "mpi_ops.h"

int main(int argc, char **argv)
{
    if (argc != 1)
    {
        printf("Wrong number of arguments.\n Exiting...\n");
        return STATUS_WRONG_INPUTS;
    }

    MPI_Init(&argc, &argv);

    MPIContext local_setup;
    set_MPIContext(&local_setup);

    MPI_Comm_size(MPI_COMM_WORLD, &(local_setup.world_size));
    MPI_Comm_rank(MPI_COMM_WORLD, &(local_setup.world_rank));

    printonrank("Successfully started.\n", 0, MPI_COMM_WORLD);

    // evaluate processor grid side
    if (local_setup.world_rank == 0)
    {
        local_setup.processor_grid_side = find_int_sqroot(local_setup.world_size);
    }

    MPI_Bcast(&local_setup.processor_grid_side, 1, MPI_INT, 0, MPI_COMM_WORLD); // fundamental to share the info with everyone; need before the division in groups
    MPI_Barrier(MPI_COMM_WORLD);
    printonrank("Broadcasted processor grid side.\n", 0, MPI_COMM_WORLD);

    // now organize the processes (manager, workers and cartesian grid)
    organize_processors(&local_setup); // TODO DESTROY COMMUNICATORS
    MPI_Barrier(MPI_COMM_WORLD);
    printonrank("Processors organized.\n", 0, MPI_COMM_WORLD);

    // for the manager, to understand the ranks
    int *full_comm_ranks = NULL; // The index is the rank inside the cartesian, the data is the rank inside the big group
    int *rank_lookup_table_cartesian = NULL;

    // open logfiles
    if (local_setup.full_group_comm != MPI_COMM_NULL)
    {
        open_logfiles(&local_setup);
        MPI_Barrier(local_setup.full_group_comm);
        printonrank("Logfiles opened successfully.\n", 0, local_setup.full_group_comm);

        obtain_rank_conversion(&local_setup, &full_comm_ranks, &rank_lookup_table_cartesian);
        MPI_Barrier(local_setup.full_group_comm);
        printonrank("Obtained rank conversion.\n", 0, local_setup.full_group_comm);
    }

    Block **block_matrix_A = NULL; // these are used only by process 0 fullrank
    Block **block_matrix_B = NULL;

    float *matrix_C_read = NULL;

    MPI_Barrier(MPI_COMM_WORLD);

    // let's make only the actual rank 0 process handle the opening, dimension detection and reading of the matrices
    if (local_setup.full_rank == 0)
    {
        // filenames assume executable is in the build directory - assume running from the "matrix_multiplication folder"
        obtain_full_matrices("./data/sample_input_matrices/matrix_a.csv", "./data/sample_input_matrices/matrix_b.csv", &local_setup, &block_matrix_A, &block_matrix_B);
        printf("Full matrices read and decomposed\n");
    }

    // MPI_Bcast(&local_setup.status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    //
    // if (local_setup.status != STATUS_OK)
    // {
    //     if (local_setup.world_rank == 0)
    //     {
    //         printf("Aborting due to an error during initialization.\n");
    //     }
    //     MPI_Finalize();
    //     return local_setup.status; // Exit with an error code.
    // }

    MPI_Barrier(MPI_COMM_WORLD);
    if (local_setup.full_group_comm != MPI_COMM_NULL)
    {
        scatter_block_dims(&local_setup, block_matrix_A);
        MPI_Barrier(local_setup.full_group_comm);
        printonrank("Dimensions of the static grid block have been scattered.\n", 0, local_setup.full_group_comm);
    }

    if (local_setup.cart_comm != MPI_COMM_NULL) // here we initialize and allocate memory only on the cartesian grid
    {
        cartesian_explorer(&local_setup);
        MPI_Barrier(local_setup.cart_comm);
        printonrank("Informations about cartesian neighbors have been distributed.\n", 0, local_setup.cart_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printonrank("Finished local_setup. Now the systolic phase starts.\n", 0, local_setup.full_group_comm);

    if (local_setup.full_group_comm != MPI_COMM_NULL)
    {
        main_loop(&local_setup, rank_lookup_table_cartesian, block_matrix_A, block_matrix_B);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (local_setup.cart_comm != MPI_COMM_NULL)
    {
        fprintf(local_setup.log_file, "rank in cart %d \n", local_setup.cart_rank);
        show_matrix(local_setup.local_block, local_setup.local_block_rows, local_setup.local_block_cols);
    }
    // Resend all the pieces to the root process

    // Receive Dims and reconstruct matrix

    if (local_setup.full_group_comm != MPI_COMM_NULL)
    {

        int local_dims[2] = {0, 0};
        if (local_setup.worker_comm != MPI_COMM_NULL)
        {
            local_dims[0] = local_setup.local_block_rows;
            local_dims[1] = local_setup.local_block_cols;
        }

        int *full_dims = {0};

        if (local_setup.full_rank == 0)
        {
            fprintf(local_setup.log_file, "Matrix side %d\n", local_setup.matrix_side);
            matrix_C_read = (float *)malloc(local_setup.matrix_side * local_setup.matrix_side * sizeof(float));
            full_dims = (int *)malloc(2 * local_setup.full_size * sizeof(int));
        }

        MPI_Gather(local_dims, 2, MPI_INT, full_dims, 2, MPI_INT, 0, local_setup.full_group_comm);

        if (local_setup.full_rank == 0)
        {
            for (int i = 0; i < local_setup.full_size; i++)
            {
                // printf("Dims sent from process %d: %d rows %d cols\n", i, full_dims[2 * i], full_dims[2 * i + 1]);
            }
        }

        int *sending_dimension = NULL;

        int *displacements = NULL;

        if (local_setup.full_rank == 0)
        {
            sending_dimension = (int *)malloc(local_setup.full_size * sizeof(int));
            displacements = (int *)malloc(local_setup.full_size * sizeof(int));

            displacements[0] = 0;
            sending_dimension[0] = 0;

            for (int i = 0; i < local_setup.processor_grid_side * local_setup.processor_grid_side; i++)
            {
                sending_dimension[i + 1] = full_dims[2 * i + 2] * full_dims[2 * i + 3];
            }

            for (int i = 0; i < local_setup.processor_grid_side * local_setup.processor_grid_side; i++)
            {

                displacements[i + 1] = displacements[i] + sending_dimension[i];
            }

            for (int i = 0; i < local_setup.processor_grid_side * local_setup.processor_grid_side + 1; i++)
            {
                printf("     %d     \n", sending_dimension[i]);
            }

            for (int i = 0; i < local_setup.processor_grid_side * local_setup.processor_grid_side + 1; i++)
            {

                printf("     %d     \n", displacements[i]);
            }
        }

        MPI_Barrier(local_setup.full_group_comm);

        MPI_Gatherv(local_setup.local_block, local_setup.local_block_cols * local_setup.local_block_rows, MPI_FLOAT, matrix_C_read, sending_dimension, displacements, MPI_FLOAT, 0, local_setup.full_group_comm);

        MPI_Barrier(local_setup.full_group_comm);

        for (int i = 0; i < local_setup.local_block_cols * local_setup.local_block_rows; i++)
        {
            fprintf(local_setup.log_file, "LOCAL BLOCK: from %d position %d:  %f\n", local_setup.full_rank, i, local_setup.local_block[i]);
        }

        MPI_Barrier(local_setup.full_group_comm);

        if (local_setup.full_rank == 0)
        {
            for (int i = 0; i < local_setup.matrix_side * local_setup.matrix_side; i++)
            {
                fprintf(local_setup.log_file, "MATRIX position %d %f \n", i, matrix_C_read[i]);
            }
        }
    }

    float *final_matrix = NULL;

    if (local_setup.full_group_comm != MPI_COMM_NULL)
    {
        if (local_setup.full_rank == 0)
        {
            final_matrix = (float *)malloc(local_setup.matrix_side * local_setup.matrix_side * sizeof(float)); // result of the multiplications
            // function REORDER THE MATRIX
            from_blocks_to_matrix(matrix_C_read, &final_matrix, local_setup.matrix_block_structure, local_setup.processor_grid_side, local_setup.matrix_side);

            // produce the output csv file

            FILE *file_to_write = NULL;
            produce_csv(&file_to_write, "./data/sample_input_matrices/matrix_c.csv", final_matrix, local_setup.matrix_side);
            fprintf(local_setup.log_file, "ziopersdsij");
        }
    }
    // End section, freeing

    if (local_setup.cart_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&local_setup.cart_comm);
    }
    if (local_setup.worker_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&local_setup.worker_comm);
    } // it needs to be put here since the process that got managers_and_workers equal to MPI_UNDEFINED didn't get their communicator initialized
    if (local_setup.full_group_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&local_setup.full_group_comm);
    }

    // free matrix of blocks
    if (block_matrix_A != NULL)
    {
        destroy_matrix_of_blocks(block_matrix_A, local_setup.processor_grid_side);
    }
    if (block_matrix_B != NULL)
    {
        destroy_matrix_of_blocks(block_matrix_B, local_setup.processor_grid_side);
    }

    // free pointers
    if (local_setup.matrix_block_structure != NULL)
    {
        free(local_setup.matrix_block_structure);
    }

    MPI_Finalize();
    return 0;
}