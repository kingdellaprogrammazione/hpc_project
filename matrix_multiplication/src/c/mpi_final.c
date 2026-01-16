#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

#include "mpi_profiler.h"
#include "mpi_ops.h"
#include "mpi_functions.h"
#include "structs.h"
#include "consts.h"
#include "io.h"

int main(int argc, char **argv)
{
    // all processor run this
    char *timestamp = NULL;

    // Accept no arguments except for timestamp
    if (argc != 1 && argc != 2)
    {
        printf("Wrong number of arguments.\n Exiting...\n");
        return STATUS_WRONG_INPUTS;
    }

    // Set up profiling for info about communication timings
    start_comm_profiling();

    double total_time = 0;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    double total_time_start = MPI_Wtime();

    // Set and declare local processor's variables to 0
    MPIContext local_setup;
    set_MPIContext(&local_setup);

    // Each processor knows the total number of processors and their rank
    MPI_Comm_size(MPI_COMM_WORLD, &(local_setup.world_size));
    MPI_Comm_rank(MPI_COMM_WORLD, &(local_setup.world_rank));

    // 0-ranked processor on the global communicator print a check message
    // TODO maybe here is needed a synchronizing moment
    printonrank("\n \nSuccessfully started MPI.\n", 0, MPI_COMM_WORLD);

    // 0-ranked processor evaluates biggest square that can be built with available processors
    if (local_setup.world_rank == 0)
    {
        local_setup.processor_grid_side = find_int_sqroot(local_setup.world_size); // TODO here implement handling of 1 process grid or exact square grids (with the solution of the double
                                                                                   // role process)
        printf("Based on the available number of processors, the biggest square grid is %dx%d.\n", local_setup.processor_grid_side, local_setup.processor_grid_side);
    }

    // 0-ranked processor broadcasts to every processor the dimension of the grid
    MPI_Bcast(&local_setup.processor_grid_side, 1, MPI_INT, 0, MPI_COMM_WORLD); // fundamental to share the info with everyone; need before the division in groups

    // Sync every processor
    MPI_Barrier(MPI_COMM_WORLD);
    printonrank("Broadcasted processor grid side.\n", 0, MPI_COMM_WORLD);

    // Organize the processes (manager, workers and cartesian grid)
    organize_processors(&local_setup); // TODO DESTROY COMMUNICATORS

    // Sync every processor
    MPI_Barrier(MPI_COMM_WORLD);
    printonrank("Processors organized.\n", 0, MPI_COMM_WORLD);

    // If timestamp was passed, everybody has it
    if (argc == 2)
    {
        timestamp = argv[1];
    }
    else // otherwise everybody allocates for the future broadcast
    {
        timestamp = (char *)malloc(20 * sizeof(char));
    }

    // The manager evaluates the timestamp and assign it to its variable, we use world_rank
    if (local_setup.world_rank == 0)
    {
        if (argc == 1)
        {
            printf("Using internal timestamps.\n");

            // Get the time to write in the file name
            time_t now = time(NULL);
            struct tm *t = localtime(&now);
            // Format: MMDD_HHMMSS (e.g., 0912_143025 for Sept 12, 14:30:25)
            strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", t);
        }
        else
        {
            printf("Taking timestamps from outside.\n");
        }
    }

    if (argc == 1)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&timestamp, sizeof(timestamp), MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // DEBUG: for the manager, to understand the ranks
    int *full_comm_ranks = NULL; // The index is the rank inside the cartesian, the data is the rank inside the big group
    int *rank_lookup_table_cartesian = NULL;

    MPI_Barrier(MPI_COMM_WORLD);

    // Open logfiles
    if (local_setup.full_group_comm != MPI_COMM_NULL)
    {
        int result = open_logfiles(&local_setup, 1);
        // TODO: handle errors

        MPI_Barrier(local_setup.full_group_comm);
        printonrank("Logfiles opened successfully.\n", 0, local_setup.full_group_comm);

        // Since the creation of the cartesian grid probably scrambled the world-cart ranks, obtain the correspondence table
        // The table is a flattened matrix where you input the coordinates and you obtain the rank in full_group
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

    // double start_time = 0;
    // double end_time = 0;

    // double start_time = MPI_Wtime();
    double distribution_computation_gathering_time = 0;
    if (local_setup.full_group_comm != MPI_COMM_NULL)
    {
        double distribution_computation_gathering_time_start = MPI_Wtime();

        main_loop(&local_setup, rank_lookup_table_cartesian, block_matrix_A, block_matrix_B);

        MPI_Barrier(local_setup.full_group_comm);

        printonrank("Systolic phase has ended succesfully.\n", 0, local_setup.full_group_comm);

        collect_and_merge(&local_setup, matrix_C_read, final_matrix);

        MPI_Barrier(local_setup.full_group_comm);
        printonrank("Construction of final matrix has been completed.\n", 0, local_setup.full_group_comm);

        double distribution_computation_gathering_time_end = MPI_Wtime();
        distribution_computation_gathering_time = distribution_computation_gathering_time_end - distribution_computation_gathering_time_start;
        if (local_setup.full_rank == 0)
        {
            free(rank_lookup_table_cartesian);
            // ONLY THE FULL_RANK 0 NEEDS TO FREE
        }
    }
    // double end_time = MPI_Wtime();

    stop_comm_profiling();
    //  print_comm_profile(local_setup.world_rank);

    double total_time_end = MPI_Wtime();
    total_time = total_time_end - total_time_start;

    MPI_Barrier(MPI_COMM_WORLD);

    // comm_profile declared as extern

    int err = save_info_timings(&local_setup, timestamp, total_time, distribution_computation_gathering_time);

    MPI_Barrier(MPI_COMM_WORLD);

    // End section, freeing
    free_all(&local_setup, block_matrix_A, block_matrix_B);
    MPI_Barrier(MPI_COMM_WORLD);
    printonrank("Freeing/closing completed.\n", 0, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}