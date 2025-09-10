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

    if (local_setup.full_group_comm != MPI_COMM_NULL) // here we scatter the individual info of each block evaluated at the beginning by process 0 between the various processes, can we put this into a function?
    {                                                 // this because we know that the cartesian order is given in the classical row major order
        int *array_cols = NULL;
        int *array_rows = NULL;

        if (local_setup.full_rank == 0)
        {
            // create the array
            array_rows = (int *)malloc((local_setup.processor_grid_side * local_setup.processor_grid_side + 1) * sizeof(int)); // allocate a long 1D array of pointers (row major) || need to include also the sending manager
            // Step 3: Check for allocation failure
            if (array_rows == NULL)
            {
                fprintf(stderr, "Failed to allocate memory.\n");
                return STATUS_ALLOCATION_FAILED; // Exit with an error
            }

            array_cols = (int *)malloc((local_setup.processor_grid_side * local_setup.processor_grid_side + 1) * sizeof(int)); // allocate a long 1D array of pointers (row major)
            // Step 3: Check for allocation failure
            if (array_cols == NULL)
            {
                fprintf(stderr, "Failed to allocate memory.\n");
                return STATUS_ALLOCATION_FAILED; // Exit with an error
            }
            array_rows[0] = 0;
            array_cols[0] = 0;

            fprintf(local_setup.log_file, "Rows in block %d are %d\n", 0, array_rows[0]);
            fprintf(local_setup.log_file, "Cols in block %d are %d\n", 0, array_cols[0]);

            for (int i = 0; i < local_setup.processor_grid_side * local_setup.processor_grid_side; i++)
            {
                array_rows[i + 1] = block_matrix_A[i]->dims[0]; // ok, corrected the +1 for the scatter
                fprintf(local_setup.log_file, "Rows in block %d are %d\n", i + 1, array_rows[i + 1]);
                array_cols[i + 1] = block_matrix_A[i]->dims[1];
                fprintf(local_setup.log_file, "Cols in block %d are %d\n", i + 1, array_cols[i + 1]);
            }
        }

        MPI_Scatter(array_rows, 1, MPI_INT, &(local_setup.local_block_rows), 1, MPI_INT, 0, local_setup.full_group_comm); // use pointer to input argument otherwise the function can't write to the outside vars
        MPI_Scatter(array_cols, 1, MPI_INT, &(local_setup.local_block_cols), 1, MPI_INT, 0, local_setup.full_group_comm);

        if (local_setup.full_rank == 0)

        {
            free(array_cols);
            free(array_rows);
        }
    }

    cartesian_explorer(&local_setup);
    MPI_Barrier(MPI_COMM_WORLD);

    printonrank("Finished local_setup. Now the systolic phase starts.\n", 0, local_setup.full_group_comm);

    if (local_setup.full_group_comm != MPI_COMM_NULL)
    {

        // set up pulses
        int pulses = local_setup.processor_grid_side * 3 - 1; // evaluate number of systolic pulses , consider also the pulses to empty the array
        // TODO RISPARMIARE TEMPO QUANDO SO CHE STO FACENDO MOLTIPLICAZIONI CON MATRICI A ZERO,
        // comunque non perdiamo tempo perche ci sono alcune moltiplicazioni che effettivamente vengono effettuate

        for (int clock = 0; clock < pulses; clock++)
        {
            fprintf(local_setup.log_file, "Process full_rank %d, cart_rank %d: entering clock %d.\n", local_setup.full_rank, local_setup.cart_rank, clock);
            // prepare the first blocks to send

            int start_index = 0; // incluso
            int end_index = 1;   // escluso

            if (local_setup.full_rank == 0)
            {

                start_index = clock - local_setup.processor_grid_side + 1; // controllare questo TODO
                if (start_index < 0)
                {
                    start_index = 0;
                }

                end_index = clock + 1;
                if (end_index > local_setup.processor_grid_side)
                {
                    end_index = local_setup.processor_grid_side; // escluso
                }

                int null_dims[2] = {0, 0};

                fprintf(local_setup.log_file, "Clock %d, Indexes edges %d %d.\n", clock, start_index, end_index);

                // send on the left side matrix A
                for (int i = 0; i < local_setup.processor_grid_side; i++)
                {
                    int rank_to_send_to = rank_lookup_table_cartesian[i * local_setup.processor_grid_side + 0];

                    if (i >= start_index && i < end_index)
                    {
                        int index_row = i;
                        int index_col = clock - i;
                        // go from coordinates to cartesian grid rank , I do not need this in reality, only for code clarity

                        MPI_Send(block_matrix_A[index_row * local_setup.processor_grid_side + index_col]->dims, 2, MPI_INT, rank_to_send_to, 0, local_setup.full_group_comm); // send info about the incoming block's dimensions

                        fprintf(local_setup.log_file, "Clock %d Process %d, full_rank %d, cart_rank %d MASTER sending the dims to proc full_rank %d\n", clock, local_setup.full_rank, local_setup.full_rank, local_setup.cart_rank, rank_to_send_to);
                    }
                    else
                    {
                        MPI_Send(null_dims, 2, MPI_INT, rank_to_send_to, 0, local_setup.full_group_comm); // send info about the incoming block's dimensions
                                                                                                          //  MPI_Send(NULL, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                                                                                                          // mandare blocco nullo  // non mando nulla
                        fprintf(local_setup.log_file, "Clock %d Process %d, full_rank %d, cart_rank %d MASTER sending the NULL DIMS to proc full_rank %d\n", clock, local_setup.full_rank, local_setup.full_rank, local_setup.cart_rank, rank_to_send_to);
                    }
                }

                // send on the top side matrix B
                for (int i = 0; i < local_setup.processor_grid_side; i++)
                {
                    int rank_to_send_to = rank_lookup_table_cartesian[0 * local_setup.processor_grid_side + i];

                    if (i >= start_index && i < end_index)
                    {
                        int index_row = clock - i;
                        int index_col = i;

                        MPI_Send(block_matrix_B[index_row * local_setup.processor_grid_side + index_col]->dims, 2, MPI_INT, rank_to_send_to, 0, local_setup.full_group_comm); // send info about the incoming block's dimensions
                        fprintf(local_setup.log_file, "Clock %d, Process %d, full_rank %d, cart_rank %d MASTER sending the dims to proc full_rank %d\n", clock, local_setup.full_rank, local_setup.full_rank, local_setup.cart_rank, rank_to_send_to);
                    }
                    else
                    {
                        MPI_Send(null_dims, 2, MPI_INT, rank_to_send_to, 0, local_setup.full_group_comm); // send info about the incoming block's dimensions
                                                                                                          //  MPI_Send(NULL, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                                                                                                          // mandare blocco nullo  // non mando nulla
                        fprintf(local_setup.log_file, "Clock %d, Process %d, full_rank %d, cart_rank %d MASTER sending the NULL DIMS to proc full_rank %d\n", clock, local_setup.full_rank, local_setup.full_rank, local_setup.cart_rank, rank_to_send_to);
                    }
                }
            }

            MPI_Barrier(local_setup.full_group_comm);
            fprintf(local_setup.log_file, "Clock %d Distribution from MASTER ended. Starting the grid's work.\n", clock);

            // considering only worker grid process
            if (local_setup.cart_comm != MPI_COMM_NULL) // remember that until explicitly changed the incoming and outcoming dimensions are 0.
            {

                int block_total_elements = local_setup.local_block_rows * local_setup.local_block_cols; // local

                if ((local_setup.cartesian_coords)[1] == 0 && (local_setup.cartesian_coords)[0] == 0) // on the corner
                {

                    // first check to have received a non zero block
                    MPI_Recv(local_setup.coming_block_dims_horizontal, 2, MPI_INT, 0, 0, local_setup.full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator

                    MPI_Recv(local_setup.coming_block_dims_vertical, 2, MPI_INT, 0, 0, local_setup.full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator

                    fprintf(local_setup.log_file, "Clock %d Process %d WORKER cart_rank %d receiving the DIMS from proc%d\n", clock, local_setup.full_rank, local_setup.cart_rank, 0);
                }

                // remember everyone receives from the 0 rank process
                else if ((local_setup.cartesian_coords)[1] == 0 && (local_setup.cartesian_coords)[0] != 0) // on the left border
                {
                    // first check to have received a non zero block

                    MPI_Recv(local_setup.coming_block_dims_horizontal, 2, MPI_INT, 0, 0, local_setup.full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
                    fprintf(local_setup.log_file, "Clock %d Process %d WORKER cart_rank %d receiving the horizontal DIMS %d %d from proc full_rank %d\n", clock, local_setup.full_rank, local_setup.cart_rank, local_setup.coming_block_dims_horizontal[0], local_setup.coming_block_dims_horizontal[1], 0);
                }
                else if ((local_setup.cartesian_coords)[0] == 0 && (local_setup.cartesian_coords)[1] != 0) // on the top border
                {
                    MPI_Recv(local_setup.coming_block_dims_vertical, 2, MPI_INT, 0, 0, local_setup.full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
                    fprintf(local_setup.log_file, "Clock %d Process %d WORKER cart_rank %d receiving the vertical DIMS %d %d from proc full_rank %d\n", clock, local_setup.full_rank, local_setup.cart_rank, local_setup.coming_block_dims_vertical[0], local_setup.coming_block_dims_vertical[1], 0);

                    // Send the block
                }

                // devono eseguire tutti il blocco seguente, non

                // everyone needs to execute this
                MPI_Sendrecv(                                                                                // no op if the source and objective are MPI_NULL_PROC
                    local_setup.going_block_dims_vertical, 2, MPI_INT, local_setup.sendto_vertical, 0,       // Send my data to the right
                    local_setup.coming_block_dims_vertical, 2, MPI_INT, local_setup.receivefrom_vertical, 0, // Receive new data from the left
                    local_setup.cart_comm, MPI_STATUS_IGNORE);
                fprintf(local_setup.log_file, "Clock %d, Process %d WORKER cart_rank %d send vertical DIMS %d %d to proc full_rank %d and receive from%d\n", clock, local_setup.full_rank, local_setup.cart_rank, local_setup.sendto_vertical, local_setup.receivefrom_vertical);

                MPI_Sendrecv(
                    local_setup.going_block_dims_horizontal, 2, MPI_INT, local_setup.sendto_horizontal, 0,       // Send my data to the right
                    local_setup.coming_block_dims_horizontal, 2, MPI_INT, local_setup.receivefrom_horizontal, 0, // Receive new data from the left
                    local_setup.cart_comm, MPI_STATUS_IGNORE);
                fprintf(local_setup.log_file, "Clock %d,Process %d WORKER cart_rank %d send horizontal DIMS %d %d to proc%d and receive from%d\n", clock, local_setup.full_rank, local_setup.cart_rank, local_setup.sendto_horizontal, local_setup.receivefrom_horizontal);
            }

            MPI_Barrier(local_setup.full_group_comm);

            // we've sent the dimensions, now send the blocks

            if (local_setup.full_rank == 0)
            {
                fprintf(local_setup.log_file, "Clock %d Ended dimension distribution. Block distribution starting.\n", clock);

                for (int i = 0; i < local_setup.processor_grid_side; i++)
                {
                    int rank_to_send_to = rank_lookup_table_cartesian[i * local_setup.processor_grid_side + 0];

                    if (i >= start_index && i < end_index)
                    {
                        int index_row = i;
                        int index_col = clock - i;
                        // go from coordinates to cartesian grid rank , I do not need this in reality, only for code clarity

                        int block_total_elements = block_matrix_A[index_row * local_setup.processor_grid_side + index_col]->dims[0] * block_matrix_A[index_row * local_setup.processor_grid_side + index_col]->dims[1];
                        fprintf(local_setup.log_file, "CHECK Process %d MASTER %d\n", local_setup.full_rank, block_total_elements);

                        MPI_Send(block_matrix_A[index_row * local_setup.processor_grid_side + index_col]->ptr_to_block_contents, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, local_setup.full_group_comm);
                        fprintf(local_setup.log_file, "Process %d MASTER sending the BLOCK to proc%d\n", local_setup.full_rank, rank_to_send_to);
                    }
                    else
                    {
                        //  MPI_Send(NULL, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                        // mandare blocco nullo  // non mando nulla
                        MPI_Send(NULL, 0, MPI_FLOAT, rank_to_send_to, 0, local_setup.full_group_comm);
                        fprintf(local_setup.log_file, "Process %d MASTER sending the NULL BLOCK to proc%d\n", local_setup.full_rank, rank_to_send_to);
                    }
                }

                // send on the top side matrix B
                for (int i = 0; i < local_setup.processor_grid_side; i++)
                {
                    int rank_to_send_to = rank_lookup_table_cartesian[0 * local_setup.processor_grid_side + i];

                    if (i >= start_index && i < end_index)
                    {
                        int index_row = clock - i;
                        int index_col = i;

                        int block_total_elements = block_matrix_B[index_row * local_setup.processor_grid_side + index_col]->dims[0] * block_matrix_B[index_row * local_setup.processor_grid_side + index_col]->dims[1];

                        MPI_Send(block_matrix_B[index_row * local_setup.processor_grid_side + index_col]->ptr_to_block_contents, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, local_setup.full_group_comm);
                        fprintf(local_setup.log_file, "Process %d MASTER sending the BLOCK to proc%d\n", local_setup.full_rank, rank_to_send_to);
                    }
                    else
                    {
                        //  MPI_Send(NULL, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                        // mandare blocco nullo  // non mando nulla
                        MPI_Send(NULL, 0, MPI_FLOAT, rank_to_send_to, 0, local_setup.full_group_comm);
                        fprintf(local_setup.log_file, "Process %d MASTER sending the NULL BLOCK to proc%d\n", local_setup.full_rank, rank_to_send_to);
                    }
                }
            }

            MPI_Barrier(local_setup.full_group_comm);
            fprintf(local_setup.log_file, "Ci siamo\n");

            if (local_setup.cart_comm != MPI_COMM_NULL)
            {
                int coming_horizontal_blocks_elements = local_setup.coming_block_dims_horizontal[0] * local_setup.coming_block_dims_horizontal[1];
                int coming_vertical_blocks_elements = local_setup.coming_block_dims_vertical[0] * local_setup.coming_block_dims_vertical[1]; // TODO CHECK IF WE CAN SIMPLIFY LOGIC ASSUMING THE DIM IS SAME WITH HORIZONTAL

                fprintf(local_setup.log_file, "cart_rank %d Iteration %d, coming block horizontal dim %d , coming block vertical %d\n", local_setup.cart_rank, clock, coming_horizontal_blocks_elements, coming_vertical_blocks_elements);
                // i sent the dimensions of the blocks coming and going // aggiungere invii di blocchi non dovrebbe creare problematiche in quanto si sta svolgendo tutto in parallelo.

                if (local_setup.coming_block_dims_horizontal[0] * local_setup.coming_block_dims_horizontal[1] != 0)
                {
                    local_setup.coming_block_horizontal = (float *)malloc(local_setup.coming_block_dims_horizontal[0] * local_setup.coming_block_dims_horizontal[1] * sizeof(float));
                }
                else
                {
                    local_setup.coming_block_horizontal = NULL;
                }
                if (local_setup.coming_block_dims_vertical[0] * local_setup.coming_block_dims_vertical[1] != 0)
                {
                    local_setup.coming_block_vertical = (float *)malloc(local_setup.coming_block_dims_vertical[0] * local_setup.coming_block_dims_vertical[1] * sizeof(float));
                }
                else
                {
                    local_setup.coming_block_vertical = NULL;
                }

                // if (clock == 0)
                // {
                //     local_setup.going_block_horizontal = (float *)malloc(local_setup.going_block_dims_horizontal[0] * local_setup.going_block_dims_horizontal[1] * sizeof(float));
                //     local_setup.going_block_vertical = (float *)malloc(local_setup.going_block_dims_vertical[0] * local_setup.going_block_dims_vertical[1] * sizeof(float));
                // }

                if (local_setup.cartesian_coords[1] == 0 && local_setup.cartesian_coords[0] == 0) // on the corner
                {
                    MPI_Recv(local_setup.coming_block_horizontal, coming_horizontal_blocks_elements, MPI_FLOAT, 0, 0, local_setup.full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
                    MPI_Recv(local_setup.coming_block_vertical, coming_vertical_blocks_elements, MPI_FLOAT, 0, 0, local_setup.full_group_comm, MPI_STATUS_IGNORE);     // remember here the right large communicator

                    fprintf(local_setup.log_file, "Process %d WORKER receiving the BLOCK from proc%d, %d\n", local_setup.full_rank, 0, 0);
                }

                // remember everyone receives from the 0 rank process
                else if (local_setup.cartesian_coords[1] == 0 && local_setup.cartesian_coords[0] != 0) // on the left border
                {
                    MPI_Recv(local_setup.coming_block_horizontal, coming_horizontal_blocks_elements, MPI_FLOAT, 0, 0, local_setup.full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
                    fprintf(local_setup.log_file, "Process %d WORKER receiving the BLOCK from proc%d\n", local_setup.full_rank, 0);
                }
                else if (local_setup.cartesian_coords[0] == 0 && local_setup.cartesian_coords[1] != 0) // on the top border
                {
                    MPI_Recv(local_setup.coming_block_vertical, coming_vertical_blocks_elements, MPI_FLOAT, 0, 0, local_setup.full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
                    fprintf(local_setup.log_file, "Process %d WORKER receiving the BLOCK from proc%d\n", local_setup.full_rank, 0);
                }

                // send current blocks

                MPI_Sendrecv(
                    local_setup.going_block_horizontal, (local_setup.going_block_dims_horizontal)[0] * (local_setup.going_block_dims_horizontal)[1], MPI_FLOAT, local_setup.sendto_horizontal, 0,         // Send my data to the right
                    local_setup.coming_block_horizontal, (local_setup.coming_block_dims_horizontal)[0] * (local_setup.coming_block_dims_horizontal)[1], MPI_FLOAT, local_setup.receivefrom_horizontal, 0, // Receive new data from the left
                    local_setup.cart_comm, MPI_STATUS_IGNORE);

                fprintf(local_setup.log_file, "Process %d WORKER send BLOCKS from proc%d and receive from%d\n", local_setup.full_rank, local_setup.sendto_horizontal, local_setup.receivefrom_horizontal);

                MPI_Sendrecv(
                    local_setup.going_block_vertical, (local_setup.going_block_dims_vertical)[0] * (local_setup.going_block_dims_vertical)[1], MPI_FLOAT, local_setup.sendto_vertical, 0,         // Send my data to the right
                    local_setup.coming_block_vertical, (local_setup.coming_block_dims_vertical)[0] * (local_setup.coming_block_dims_vertical)[1], MPI_FLOAT, local_setup.receivefrom_vertical, 0, // Receive new data from the left
                    local_setup.cart_comm, MPI_STATUS_IGNORE);

                fprintf(local_setup.log_file, "Process %d WORKER send BLOCKS from proc%d and receive from%d\n", local_setup.full_rank, local_setup.sendto_vertical, local_setup.receivefrom_vertical);

                // CALCULATE

                // Compute
                // TODO CHECK COMMON DIMENSION OF INCOMING BLOCKS

                MPI_Barrier(local_setup.worker_comm);

                if (local_setup.coming_block_dims_horizontal != NULL && local_setup.coming_block_dims_horizontal != NULL) // the zeros arrive simultaneously
                {

                    matrix_multi(local_setup.coming_block_horizontal, local_setup.coming_block_vertical, local_setup.multi_result, local_setup.coming_block_dims_horizontal[0], local_setup.coming_block_dims_horizontal[1], local_setup.coming_block_dims_vertical[1]);

                    if (local_setup.cart_rank == 0)
                    {
                        fprintf(local_setup.log_file, "Clock %d +-+-+-+--+ \n", clock);
                        show_matrix(local_setup.coming_block_horizontal, local_setup.coming_block_dims_horizontal[0], local_setup.coming_block_dims_horizontal[1]);
                        fprintf(local_setup.log_file, "Clock %d \n", rand());

                        show_matrix(local_setup.coming_block_vertical, local_setup.coming_block_dims_vertical[0], local_setup.coming_block_dims_vertical[1]);
                        fprintf(local_setup.log_file, "Clock %d \n", rand());

                        show_matrix(local_setup.multi_result, local_setup.coming_block_dims_horizontal[0], local_setup.coming_block_dims_vertical[1]);
                    }
                    // test
                    float *temp = NULL;
                    temp = (float *)malloc(local_setup.local_block_cols * local_setup.local_block_rows * sizeof(float));

                    matrix_add(local_setup.multi_result, local_setup.local_block, temp, local_setup.local_block_rows, local_setup.local_block_cols); // attention here TODO check
                    local_setup.local_block = temp;
                    if (local_setup.cart_rank == 0)
                    {
                        fprintf(local_setup.log_file, "--------------------------\n");
                        show_matrix(local_setup.local_block, local_setup.local_block_rows, local_setup.local_block_cols);
                        fprintf(local_setup.log_file, "--------------------------\n");
                    }
                }
                // create zero blocks per sicurezza
                // local_setup.multi_result = create_zero_matrix(local_setup.local_block_cols * local_setup.local_block_rows);

                // update blocks and dims

                local_setup.going_block_dims_horizontal[0] = local_setup.coming_block_dims_horizontal[0];
                local_setup.going_block_dims_vertical[0] = local_setup.coming_block_dims_vertical[0];
                local_setup.going_block_dims_horizontal[1] = local_setup.coming_block_dims_horizontal[1];
                local_setup.going_block_dims_vertical[1] = local_setup.coming_block_dims_vertical[1];

                if (local_setup.going_block_horizontal != NULL)
                {
                    free(local_setup.going_block_horizontal);
                }
                if (local_setup.going_block_vertical != NULL)
                {
                    free(local_setup.going_block_vertical);
                }
                local_setup.going_block_vertical = local_setup.coming_block_vertical;
                local_setup.going_block_horizontal = local_setup.coming_block_horizontal;

                local_setup.coming_block_vertical = NULL;
                local_setup.coming_block_horizontal = NULL;

                if (local_setup.multi_result != NULL)
                {
                    free(local_setup.multi_result);
                    local_setup.multi_result = create_zero_matrix(local_setup.local_block_cols * local_setup.local_block_rows);
                }
            }

            // /// debuggggg
            //  if (local_setup.full_group_comm != MPI_COMM_NULL)
            // {
            //      log_MPIContext(&local_setup);
            //
            //     fclose(local_setup.log_file);
            // }
            //
            //  MPI_Barrier(MPI_COMM_WORLD);
            //  MPI_Abort(MPI_COMM_WORLD, 123);
            // /// debuggggg
        }

        fprintf(local_setup.log_file, "Ziopers\n");
        // CALCULATE
        MPI_Barrier(local_setup.full_group_comm);
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