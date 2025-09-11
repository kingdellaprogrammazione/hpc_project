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

    printonrank("\n \nSuccessfully started.\n", 0, MPI_COMM_WORLD);

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
        int result = open_logfiles(&local_setup, 1);
        // todo handle errors

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
        int result = obtain_full_matrices("./data/sample_input_matrices/matrix_a.csv", "./data/sample_input_matrices/matrix_b.csv", &local_setup, &block_matrix_A, &block_matrix_B);
        if (result != STATUS_OK)
        {

            // tODO
        }
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

    float *final_matrix = NULL;

    if (local_setup.full_group_comm != MPI_COMM_NULL)
    {
        main_loop(&local_setup, rank_lookup_table_cartesian, block_matrix_A, block_matrix_B);
        MPI_Barrier(local_setup.full_group_comm);
        printonrank("Systolic phase has ended succesfully.\n", 0, local_setup.full_group_comm);

        collect_and_merge(&local_setup, matrix_C_read, final_matrix);
        MPI_Barrier(local_setup.full_group_comm);
        printonrank("Construction of final matrix has been completed.\n", 0, local_setup.full_group_comm);
    }
    // End section, freeing

    free_all(&local_setup, block_matrix_A, block_matrix_B);
    MPI_Barrier(MPI_COMM_WORLD);
    printonrank("Freeing/closing completed.\n", 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}