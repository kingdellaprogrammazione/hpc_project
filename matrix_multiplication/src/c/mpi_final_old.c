#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi_functions.h"
#include "structs.h"
#include <unistd.h>
#include "consts.h"
#include <assert.h>

int main(int argc, char **argv)
{
    int number_of_input_arguments = argc;
    if (argc != 1)
    {
        printf("Wrong number of arguments.\n Exiting...\n");
        return STATUS_WRONG_INPUTS;
    }

    int world_rank;
    int world_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int processor_grid_side = 0;

    int status = STATUS_OK;

    int *matrix_block_structure = NULL;

    float *matrix_A_read = NULL; // forse queste metterle solo dentro world == 0
    float *matrix_B_read = NULL;
    float *matrix_C_read = NULL;

    Block **block_matrix_A = NULL;
    Block **block_matrix_B = NULL;

    // set up the topology
    int sendto_vertical = -1;
    int sendto_horizontal = -1;
    int receivefrom_vertical = -1;
    int receivefrom_horizontal = -1;

    // Each process will have its OWN local block of the matrices.   COMING AND GOING ARE THE SAME BLOCKS
    float *coming_block_vertical = NULL;
    float *going_block_horizontal = NULL;
    float *going_block_vertical = NULL;
    float *coming_block_horizontal = NULL;

    float *multi_result = NULL;
    float *local_block = NULL;

    int local_block_rows = 0;
    int local_block_cols = 0;

    int coming_block_dims_vertical[2] = {0};
    int coming_block_dims_horizontal[2] = {0};

    int going_block_dims_vertical[2] = {0};
    int going_block_dims_horizontal[2] = {0};

    int matrix_side = 0;

    // let's make only the actual rank 0 process handle the opening, dimension detection and reading of the matrices
    if (world_rank == 0)
    {
        // filenames assume executable is in the build directory - assume running from the "matrix_multiplication folder"
        status = obtain_full_matrices("./data/sample_input_matrices/matrix_a.csv", "./data/sample_input_matrices/matrix_b.csv", &matrix_A_read, &matrix_B_read, &matrix_side);

        // print matrix A
        show_original_matrix(matrix_A_read, matrix_side);

        // print matrix B
        show_original_matrix(matrix_B_read, matrix_side);

        // we have world_size processes available. find the nearest square integer.
        processor_grid_side = find_int_sqroot(world_size);
        printf("processor_grid_size %d\n", processor_grid_side);

        // now divide the matrix in block and do the systolic array multiplications in blocks.
        int block_size = matrix_side / processor_grid_side;
        int block_reminder = matrix_side % processor_grid_side;

        printf("block size %d\nblock reminder %d\n", block_size, block_reminder);

        // analyze matrix A (and so B) structure
        matrix_block_structure = calculate_blocks_sizes(processor_grid_side, matrix_side);

        for (int i = 0; i < processor_grid_side; i++)
        {
            printf("%d\n", matrix_block_structure[i]);
        }

        // DEBUG
        printf("Block dimensions ");
        for (int i = 0; i < processor_grid_side; i++)
        {
            printf("%d ", matrix_block_structure[i]);
        }
        printf("\n");

        // divide matrix A in blocks according to the processor grid.
        block_matrix_A = matrix_parser_general(matrix_A_read, matrix_block_structure, processor_grid_side);

        // show a block
        show_blocks_general(block_matrix_A, 0, 1, processor_grid_side);

        // divide matrix B in blocks according to the processor grid.
        block_matrix_B = matrix_parser_general(matrix_B_read, matrix_block_structure, processor_grid_side);
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (status != STATUS_OK)
    {
        if (world_rank == 0)
        {
            printf("Aborting due to an error during initialization.\n");
        }
        MPI_Finalize();
        return status; // Exit with an error code.
    }

    MPI_Bcast(&processor_grid_side, 1, MPI_INT, 0, MPI_COMM_WORLD); // fundamental to share the info with everyone; need before the division in groups

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0)
    {
        printf("Reading done\n");
    }

    MPI_Comm full_group_comm = MPI_COMM_NULL;
    MPI_Comm worker_comm = MPI_COMM_NULL;
    MPI_Comm cart_comm = MPI_COMM_NULL;

    int color = 1;

    if (processor_grid_side * processor_grid_side != world_size)
    {
        if ((world_rank > 0) && (world_rank < (processor_grid_side * processor_grid_side + 1)))
        {
            color = 1;
        }
        else
        {
            color = MPI_UNDEFINED;
        }
    }

    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &worker_comm); // edge case of square world size handled

    int cartesian_grid_rank = 0;
    int cartesian_coords[2] = {0};

    if (worker_comm != MPI_COMM_NULL) // consider only workers
    {
        // Create the Cartesian grid from the new, smaller communicator
        const int ndims = 2;
        int dims[2] = {processor_grid_side, processor_grid_side};
        int periods[2] = {0, 0};
        int reorder = 1;

        MPI_Cart_create(worker_comm, ndims, dims, periods, reorder, &cart_comm);

        // --- Now, use the grid communicator for your work ---
        MPI_Comm_rank(cart_comm, &cartesian_grid_rank);
        MPI_Cart_coords(cart_comm, cartesian_grid_rank, ndims, cartesian_coords);

        printf("World Rank %2d: ACTIVE -> Grid Rank %2d -> Coords (%d, %d)\n",
               world_rank, cartesian_grid_rank, cartesian_coords[0], cartesian_coords[1]);
    }

    if (world_rank == 0)
    {
        color = 1;
    }

    MPI_Comm_split(MPI_COMM_WORLD, color, world_rank, &full_group_comm); // this split includes all processes, also the manager, should work also in the edge case

    MPI_Barrier(MPI_COMM_WORLD);

    if (full_group_comm != MPI_COMM_NULL) // here we scatter the individual info of each block evaluated at the beginning by process 0 between the various processes, can we put this into a function?
    {                                     // this because we know that the cartesian order is given in the classical row major order
        int *array_cols = NULL;
        int *array_rows = NULL;

        if (world_rank == 0)
        {
            // create the array
            array_rows = (int *)malloc((processor_grid_side * processor_grid_side + 1) * sizeof(int)); // allocate a long 1D array of pointers (row major) || need to include also the sending manager
            // Step 3: Check for allocation failure
            if (array_rows == NULL)
            {
                fprintf(stderr, "Failed to allocate memory.\n");
                return STATUS_ALLOCATION_FAILED; // Exit with an error
            }

            array_cols = (int *)malloc((processor_grid_side * processor_grid_side + 1) * sizeof(int)); // allocate a long 1D array of pointers (row major)
            // Step 3: Check for allocation failure
            if (array_cols == NULL)
            {
                fprintf(stderr, "Failed to allocate memory.\n");
                return STATUS_ALLOCATION_FAILED; // Exit with an error
            }
            array_rows[0] = 0;
            array_cols[0] = 0;

            for (int i = 0; i < processor_grid_side * processor_grid_side; i++)
            {
                array_rows[i + 1] = block_matrix_A[i]->dims[0]; // ok, corrected the +1 for the scatter
                printf("Rows in block %d are %d\n", i, array_rows[i]);
                array_cols[i + 1] = block_matrix_A[i]->dims[1];
                printf("Cols in block %d are %d\n", i, array_cols[i]);
            }
        }
        MPI_Scatter(array_rows, 1, MPI_INT, &local_block_rows, 1, MPI_INT, 0, full_group_comm); // use pointer to input argument otherwise the function can't write to the outside vars
        MPI_Scatter(array_cols, 1, MPI_INT, &local_block_cols, 1, MPI_INT, 0, full_group_comm);

        int full_group_rank = 0;
        MPI_Comm_rank(full_group_comm, &full_group_rank);

        if (world_rank == 0)
        {
            free(array_cols);
            free(array_rows);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (cart_comm != MPI_COMM_NULL) // here we initialize and allocate memory only on the cartesian grid
    {
        printf("Cartesian Rank %d Rows %d Cols %d \n", cartesian_grid_rank, local_block_rows, local_block_cols);
        int block_total_elements = local_block_rows * local_block_cols; // individual result for each block

        multi_result = (float *)malloc(block_total_elements * sizeof(float)); // result of the multiplications
        local_block = (float *)malloc(block_total_elements * sizeof(float));  // locally hosted block

        assert(block_total_elements != 0);
        if (multi_result == NULL || local_block == NULL)
        {
            printf("P%d: Failed to allocate memory for local blocks.\n", world_rank);
            MPI_Abort(MPI_COMM_WORLD, 1); // A hard stop for errors
        }

        // Discover vertical neighbors
        MPI_Cart_shift(cart_comm, 0, 1, &receivefrom_vertical, &sendto_vertical);

        // Discover horizontal neighbors
        MPI_Cart_shift(cart_comm, 1, 1, &receivefrom_horizontal, &sendto_horizontal);

        multi_result = create_zero_matrix(block_total_elements);
        local_block = create_zero_matrix(block_total_elements);
        printf("++++++++++++++++++++++++++++++\n");

        // show_matrix(local_block, local_block_rows, local_block_cols);
        printf("++++++++++++++++++++++++++++++\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (world_rank == 0)
    {
        printf("Ci srrivo");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int cart_rank = -1;
    int cart_size = 0;
    int full_rank = -1;
    int full_size = 0;

    if (full_group_comm != MPI_COMM_NULL)
    {
        MPI_Comm_size(full_group_comm, &full_size);
        MPI_Comm_rank(full_group_comm, &full_rank);
    }
    if (cart_comm != MPI_COMM_NULL)
    {
        MPI_Comm_size(cart_comm, &cart_size);
        MPI_Comm_rank(cart_comm, &cart_rank);
    }

    // for the manager, to understand the ranks
    int *full_comm_ranks = NULL; // The index is the rank inside the cartesian, the data is the rank inside the big group
    int *rank_lookup_table_cartesian = NULL;

    if (full_group_comm != MPI_COMM_NULL)
    {
        int *full_comm_ranks = malloc(cart_size * sizeof(int));

        int cart_rank_in_full_world = -1;
        // We use an Allreduce to find the rank of the process where cart_rank is 0.
        // The process with cart_rank == 0 contributes its full_rank. Everyone else contributes -1.
        // The MPI_MAX operation will find the one valid rank.
        int root_candidate = (cart_rank == 0) ? full_rank : -1; // the case for cart_rank  == -1 is automatically escluded

        MPI_Allreduce(&root_candidate, &cart_rank_in_full_world, 1, MPI_INT, MPI_MAX, full_group_comm);

        if (cart_rank == 0) // this needs to be executed from a process that belongs to both groups
        {

            // STEP 2: Translate this rank into even_comm
            MPI_Group cart_group, full_group;
            MPI_Comm_group(cart_comm, &cart_group);
            MPI_Comm_group(full_group_comm, &full_group);

            int rank_to_translate[cart_size];
            for (int i = 0; i < cart_size; i++)
            {
                rank_to_translate[i] = i;
            }

            MPI_Group_translate_ranks(cart_group, cart_size, rank_to_translate, full_group, full_comm_ranks);
            for (int i = 0; i < cart_size; i++)
            {
                printf("Rank %d in cartesian is rank %d in bigger group\n", i, full_comm_ranks[i]);
            }
            // Clean up groups
            MPI_Group_free(&cart_group);
            MPI_Group_free(&full_group);

            MPI_Bcast(full_comm_ranks, cart_size, MPI_INT, cart_rank_in_full_world, full_group_comm);

            rank_lookup_table_cartesian = malloc(processor_grid_side * processor_grid_side * sizeof(int));
            for (int i = 0; i < processor_grid_side; i++)
            {
                for (int j = 0; j < processor_grid_side; j++)
                {
                    int temp_coords[2] = {i, j};
                    int temp_cart_rank;
                    // This call is legal here because this process is in cart_comm!
                    MPI_Cart_rank(cart_comm, temp_coords, &temp_cart_rank);
                    // Use the translated ranks to find the final destination rank.
                    rank_lookup_table_cartesian[i * processor_grid_side + j] = full_comm_ranks[temp_cart_rank];
                }
            }
            MPI_Send(rank_lookup_table_cartesian, processor_grid_side * processor_grid_side, MPI_INT, 0, 99, full_group_comm);
        }
        if (full_rank == 0)
        {
            rank_lookup_table_cartesian = malloc(processor_grid_side * processor_grid_side * sizeof(int));
            // If the manager IS the root worker (e.g., 1 proc), it doesn't need to Recv.

            MPI_Recv(rank_lookup_table_cartesian, processor_grid_side * processor_grid_side, MPI_INT, cart_rank_in_full_world, 99, full_group_comm, MPI_STATUS_IGNORE);

            printf("Manager has received the rank lookup table.\n");
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0)
    {
        printf("Finished setup. Now the systolic phase starts.\n");
    }

    if (full_group_comm != MPI_COMM_NULL)
    {

        // set up pulses
        int pulses = processor_grid_side * 3 - 1; // evaluate number of systolic pulses , consider also the pulses to empty the array
        // TODO RISPARMIARE TEMPO QUANDO SO CHE STO FACENDO MOLTIPLICAZIONI CON MATRICI A ZERO,
        // comunque non perdiamo tempo perche ci sono alcune moltiplicazioni che effettivamente vengono effettuate

        for (int clock = 0; clock < pulses; clock++)
        {
            printf("Process world_rank %d, full_rank %d, cart_rank %d: entering clock %d.\n", world_rank, full_rank, cart_rank, clock);
            // prepare the first blocks to send

            int start_index = 0; // incluso
            int end_index = 1;   // escluso

            if (full_rank == 0)
            {

                start_index = clock - processor_grid_side + 1; // controllare questo TODO
                if (start_index < 0)
                {
                    start_index = 0;
                }

                end_index = clock + 1;
                if (end_index > processor_grid_side)
                {
                    end_index = processor_grid_side; // escluso
                }

                int null_dims[2] = {0, 0};

                printf("Clock %d, Indexes edges %d %d.\n", clock, start_index, end_index);

                // send on the left side matrix A
                for (int i = 0; i < processor_grid_side; i++)
                {
                    int rank_to_send_to = rank_lookup_table_cartesian[i * processor_grid_side + 0];

                    if (i >= start_index && i < end_index)
                    {
                        int index_row = i;
                        int index_col = clock - i;
                        // go from coordinates to cartesian grid rank , I do not need this in reality, only for code clarity

                        int block_total_elements = block_matrix_A[index_row * processor_grid_side + index_col]->dims[0] * block_matrix_A[index_row * processor_grid_side + index_col]->dims[0];

                        MPI_Send(block_matrix_A[index_row * processor_grid_side + index_col]->dims, 2, MPI_INT, rank_to_send_to, 0, full_group_comm); // send info about the incoming block's dimensions

                        printf("Clock %d Process %d, full_rank %d, cart_rank %d MASTER sending the dims to proc full_rank %d\n", clock, world_rank, full_rank, cart_rank, rank_to_send_to);
                    }
                    else
                    {
                        MPI_Send(null_dims, 2, MPI_INT, rank_to_send_to, 0, full_group_comm); // send info about the incoming block's dimensions
                                                                                              //  MPI_Send(NULL, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                                                                                              // mandare blocco nullo  // non mando nulla
                        printf("Clock %d Process %d, full_rank %d, cart_rank %d MASTER sending the NULL DIMS to proc full_rank %d\n", clock, world_rank, full_rank, cart_rank, rank_to_send_to);
                    }
                }

                // send on the top side matrix B
                for (int i = 0; i < processor_grid_side; i++)
                {
                    int rank_to_send_to = rank_lookup_table_cartesian[0 * processor_grid_side + i];

                    if (i >= start_index && i < end_index)
                    {
                        int index_row = clock - i;
                        int index_col = i;

                        int block_total_elements = block_matrix_B[index_row * processor_grid_side + index_col]->dims[0] * block_matrix_B[index_row * processor_grid_side + index_col]->dims[0];

                        MPI_Send(block_matrix_B[index_row * processor_grid_side + index_col]->dims, 2, MPI_INT, rank_to_send_to, 0, full_group_comm); // send info about the incoming block's dimensions
                        printf("Clock %d, Process %d, full_rank %d, cart_rank %d MASTER sending the dims to proc full_rank %d\n", clock, world_rank, full_rank, cart_rank, rank_to_send_to);
                    }
                    else
                    {
                        MPI_Send(null_dims, 2, MPI_INT, rank_to_send_to, 0, full_group_comm); // send info about the incoming block's dimensions
                                                                                              //  MPI_Send(NULL, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                                                                                              // mandare blocco nullo  // non mando nulla
                        printf("Clock %d, Process %d, full_rank %d, cart_rank %d MASTER sending the NULL DIMS to proc full_rank %d\n", clock, world_rank, full_rank, cart_rank, rank_to_send_to);
                    }
                }
            }

            MPI_Barrier(full_group_comm);
            if (full_rank == 0)
            {
                printf("Clock %d Distribution from MASTER ended. Starting the grid's work.", clock);
            }

            // considering only worker grid process
            if (cart_comm != MPI_COMM_NULL) // remember that until explicitly changed the incoming and outcoming dimensions are 0.
            {

                int block_total_elements = local_block_rows * local_block_cols; // local

                if (cartesian_coords[1] == 0 && cartesian_coords[0] == 0) // on the corner
                {

                    // first check to have received a non zero block
                    MPI_Recv(coming_block_dims_horizontal, 2, MPI_INT, 0, 0, full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator

                    MPI_Recv(coming_block_dims_vertical, 2, MPI_INT, 0, 0, full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator

                    printf("Clock %d Process %d WORKER cart_rank %d receiving the DIMS from proc%d\n", clock, world_rank, cart_rank, 0);
                }

                // remember everyone receives from the 0 rank process
                else if (cartesian_coords[1] == 0 && cartesian_coords[0] != 0) // on the left border
                {
                    // first check to have received a non zero block

                    MPI_Recv(coming_block_dims_horizontal, 2, MPI_INT, 0, 0, full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
                    printf("Clock %d Process %d WORKER cart_rank %d receiving the horizontal DIMS %d %d from proc full_rank %d\n", clock, world_rank, cart_rank, coming_block_dims_horizontal[0], coming_block_dims_horizontal[1], 0);
                }
                else if (cartesian_coords[0] == 0 && cartesian_coords[1] != 0) // on the top border
                {
                    MPI_Recv(coming_block_dims_vertical, 2, MPI_INT, 0, 0, full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
                    printf("Clock %d Process %d WORKER cart_rank %d receiving the vertical DIMS %d %d from proc full_rank %d\n", clock, world_rank, cart_rank, coming_block_dims_vertical[0], coming_block_dims_vertical[1], 0);

                    // Send the block
                }

                // devono eseguire tutti il blocco seguente, non

                // everyone needs to execute this
                MPI_Sendrecv(                                                        // no op if the source and objective are MPI_NULL_PROC
                    going_block_dims_vertical, 2, MPI_INT, sendto_vertical, 0,       // Send my data to the right
                    coming_block_dims_vertical, 2, MPI_INT, receivefrom_vertical, 0, // Receive new data from the left
                    cart_comm, MPI_STATUS_IGNORE);
                printf("Clock %d, Process %d WORKER cart_rank %d send vertical DIMS %d %d to proc full_rank %d and receive from%d\n", clock, world_rank, cart_rank, sendto_vertical, receivefrom_vertical);

                MPI_Sendrecv(
                    going_block_dims_horizontal, 2, MPI_INT, sendto_horizontal, 0,       // Send my data to the right
                    coming_block_dims_horizontal, 2, MPI_INT, receivefrom_horizontal, 0, // Receive new data from the left
                    cart_comm, MPI_STATUS_IGNORE);
                printf("Clock %d,Process %d WORKER cart_rank %d send horizontal DIMS %d %d to proc%d and receive from%d\n", clock, world_rank, cart_rank, sendto_horizontal, receivefrom_horizontal);
            }

            MPI_Barrier(full_group_comm);
            // we've sent the dimensions, now send the blocks

            if (full_rank == 0)
            {
                printf("Clock %d Ended dimension distribution. Block distribution starting.\n", clock);

                for (int i = 0; i < processor_grid_side; i++)
                {
                    int rank_to_send_to = rank_lookup_table_cartesian[i * processor_grid_side + 0];

                    if (i >= start_index && i < end_index)
                    {
                        int index_row = i;
                        int index_col = clock - i;
                        // go from coordinates to cartesian grid rank , I do not need this in reality, only for code clarity

                        int block_total_elements = block_matrix_A[index_row * processor_grid_side + index_col]->dims[0] * block_matrix_A[index_row * processor_grid_side + index_col]->dims[1];

                        MPI_Send(block_matrix_A[index_row * processor_grid_side + index_col]->ptr_to_block_contents, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                        printf("Process %d MASTER sending the BLOCK to proc%d\n", world_rank, rank_to_send_to);
                    }
                    else
                    {
                        //  MPI_Send(NULL, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                        // mandare blocco nullo  // non mando nulla
                        MPI_Send(NULL, 0, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                        printf("Process %d MASTER sending the NULL BLOCK to proc%d\n", world_rank, rank_to_send_to);
                    }
                }

                // send on the top side matrix B
                for (int i = 0; i < processor_grid_side; i++)
                {
                    int rank_to_send_to = rank_lookup_table_cartesian[0 * processor_grid_side + i];

                    if (i >= start_index && i < end_index)
                    {
                        int index_row = clock - i;
                        int index_col = i;

                        int block_total_elements = block_matrix_B[index_row * processor_grid_side + index_col]->dims[0] * block_matrix_B[index_row * processor_grid_side + index_col]->dims[1];

                        MPI_Send(block_matrix_B[index_row * processor_grid_side + index_col]->ptr_to_block_contents, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                        printf("Process %d MASTER sending the BLOCK to proc%d\n", world_rank, rank_to_send_to);
                    }
                    else
                    {
                        //  MPI_Send(NULL, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                        // mandare blocco nullo  // non mando nulla
                        MPI_Send(NULL, 0, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                        printf("Process %d MASTER sending the NULL BLOCK to proc%d\n", world_rank, rank_to_send_to);
                    }
                }
            }

            MPI_Barrier(full_group_comm);
            printf("Ci siamo\n");

            if (cart_comm != MPI_COMM_NULL)
            {
                int coming_horizontal_blocks_elements = coming_block_dims_horizontal[0] * coming_block_dims_horizontal[1];
                int coming_vertical_blocks_elements = coming_block_dims_vertical[0] * coming_block_dims_vertical[1]; // TODO CHECK IF WE CAN SIMPLIFY LOGIC ASSUMING THE DIM IS SAME WITH HORIZONTAL

                printf("cart_rank %d Iteration %d, coming block horizontal dim %d , coming block vertical %d\n", cart_rank, clock, coming_horizontal_blocks_elements, coming_vertical_blocks_elements);
                // i sent the dimensions of the blocks coming and going // aggiungere invii di blocchi non dovrebbe creare problematiche in quanto si sta svolgendo tutto in parallelo.

                if (coming_block_dims_horizontal[0] * coming_block_dims_horizontal[1] != 0)
                {
                    coming_block_horizontal = (float *)malloc(coming_block_dims_horizontal[0] * coming_block_dims_horizontal[1] * sizeof(float));
                }
                else
                {
                    coming_block_horizontal = NULL;
                }
                if (coming_block_dims_vertical[0] * coming_block_dims_vertical[1] != 0)
                {
                    coming_block_vertical = (float *)malloc(coming_block_dims_vertical[0] * coming_block_dims_vertical[1] * sizeof(float));
                }
                else
                {
                    coming_block_vertical = NULL;
                }

                // if (clock == 0)
                // {
                //     going_block_horizontal = (float *)malloc(going_block_dims_horizontal[0] * going_block_dims_horizontal[1] * sizeof(float));
                //     going_block_vertical = (float *)malloc(going_block_dims_vertical[0] * going_block_dims_vertical[1] * sizeof(float));
                // }

                if (cartesian_coords[1] == 0 && cartesian_coords[0] == 0) // on the corner
                {
                    MPI_Recv(coming_block_horizontal, coming_horizontal_blocks_elements, MPI_FLOAT, 0, 0, full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
                    MPI_Recv(coming_block_vertical, coming_vertical_blocks_elements, MPI_FLOAT, 0, 0, full_group_comm, MPI_STATUS_IGNORE);     // remember here the right large communicator

                    printf("Process %d WORKER receiving the BLOCK from proc%d, %d\n", world_rank, 0, 0);
                }

                // remember everyone receives from the 0 rank process
                else if (cartesian_coords[1] == 0 && cartesian_coords[0] != 0) // on the left border
                {
                    MPI_Recv(coming_block_horizontal, coming_horizontal_blocks_elements, MPI_FLOAT, 0, 0, full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
                    printf("Process %d WORKER receiving the BLOCK from proc%d\n", world_rank, 0);
                }
                else if (cartesian_coords[0] == 0 && cartesian_coords[1] != 0) // on the top border
                {
                    MPI_Recv(coming_block_vertical, coming_vertical_blocks_elements, MPI_FLOAT, 0, 0, full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
                    printf("Process %d WORKER receiving the BLOCK from proc%d\n", world_rank, 0);
                }

                // send current blocks

                MPI_Sendrecv(
                    going_block_horizontal, going_block_dims_horizontal[0] * going_block_dims_horizontal[1], MPI_FLOAT, sendto_horizontal, 0,         // Send my data to the right
                    coming_block_horizontal, coming_block_dims_horizontal[0] * coming_block_dims_horizontal[1], MPI_FLOAT, receivefrom_horizontal, 0, // Receive new data from the left
                    cart_comm, MPI_STATUS_IGNORE);

                printf("Process %d WORKER send BLOCKS from proc%d and receive from%d\n", world_rank, sendto_horizontal, receivefrom_horizontal);

                MPI_Sendrecv(
                    going_block_vertical, going_block_dims_vertical[0] * going_block_dims_vertical[1], MPI_FLOAT, sendto_vertical, 0,         // Send my data to the right
                    coming_block_vertical, coming_block_dims_vertical[0] * coming_block_dims_vertical[1], MPI_FLOAT, receivefrom_vertical, 0, // Receive new data from the left
                    cart_comm, MPI_STATUS_IGNORE);

                printf("Process %d WORKER send BLOCKS from proc%d and receive from%d\n", world_rank, sendto_vertical, receivefrom_vertical);

                // CALCULATE

                // Compute
                // TODO CHECK COMMON DIMENSION OF INCOMING BLOCKS

                MPI_Barrier(worker_comm);

                if (coming_block_dims_horizontal != NULL && coming_block_dims_horizontal != NULL) // the zeros arrive simultaneously
                {

                    matrix_multi(coming_block_horizontal, coming_block_vertical, multi_result, coming_block_dims_horizontal[0], coming_block_dims_horizontal[1], coming_block_dims_vertical[1]);

                    if (cart_rank == 0)
                    {
                        printf("Clock %d +-+-+-+--+ \n", clock);
                        show_matrix(coming_block_horizontal, coming_block_dims_horizontal[0], coming_block_dims_horizontal[1]);
                        printf("Clock %d \n", rand());

                        show_matrix(coming_block_vertical, coming_block_dims_vertical[0], coming_block_dims_vertical[1]);
                        printf("Clock %d \n", rand());

                        show_matrix(multi_result, coming_block_dims_horizontal[0], coming_block_dims_vertical[1]);
                    }
                    // test
                    float *temp = NULL;
                    temp = (float *)malloc(local_block_cols * local_block_rows * sizeof(float));

                    matrix_add(multi_result, local_block, temp, local_block_rows, local_block_cols); // attention here TODO check
                    local_block = temp;
                    if (cart_rank == 0)
                    {
                        printf("--------------------------\n");
                        show_matrix(local_block, local_block_rows, local_block_cols);
                        printf("--------------------------\n");
                    }
                }
                // create zero blocks per sicurezza
                // multi_result = create_zero_matrix(local_block_cols * local_block_rows);

                // update blocks and dims

                going_block_dims_horizontal[0] = coming_block_dims_horizontal[0];
                going_block_dims_vertical[0] = coming_block_dims_vertical[0];
                going_block_dims_horizontal[1] = coming_block_dims_horizontal[1];
                going_block_dims_vertical[1] = coming_block_dims_vertical[1];

                if (going_block_horizontal != NULL)
                {
                    free(going_block_horizontal);
                }
                if (going_block_vertical != NULL)
                {
                    free(going_block_vertical);
                }
                going_block_vertical = coming_block_vertical;
                going_block_horizontal = coming_block_horizontal;

                coming_block_vertical = NULL;
                coming_block_horizontal = NULL;

                if (multi_result != NULL)
                {
                    free(multi_result);
                    multi_result = create_zero_matrix(local_block_cols * local_block_rows);
                }
            }
        }

        printf("Ziopers\n");
        // CALCULATE
        MPI_Barrier(full_group_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (cart_comm != MPI_COMM_NULL)
    {
        printf("rank in cart %d \n", cart_rank);
        show_matrix(local_block, local_block_rows, local_block_cols);
    }
    // Resend all the pieces to the root process

    // Receive Dims and reconstruct matrix

    if (full_group_comm != MPI_COMM_NULL)
    {

        int local_dims[2] = {0, 0};
        if (worker_comm != MPI_COMM_NULL)
        {
            local_dims[0] = local_block_rows;
            local_dims[1] = local_block_cols;
        }

        int *full_dims = {0};

        if (full_rank == 0)
        {
            printf("Matrix side %d\n", matrix_side);
            matrix_C_read = (float *)malloc(matrix_side * matrix_side * sizeof(float));
            full_dims = (int *)malloc(2 * full_size * sizeof(int));
        }

        MPI_Gather(local_dims, 2, MPI_INT, full_dims, 2, MPI_INT, 0, full_group_comm);

        if (world_rank == 0)
        {
            for (int i = 0; i < full_size; i++)
            {
                // printf("Dims sent from process %d: %d rows %d cols\n", i, full_dims[2 * i], full_dims[2 * i + 1]);
            }
        }

        int *sending_dimension = NULL;

        int *displacements = NULL;

        if (world_rank == 0)
        {
            sending_dimension = (int *)malloc(full_size * sizeof(int));
            displacements = (int *)malloc(full_size * sizeof(int));

            displacements[0] = 0;
            sending_dimension[0] = 0;

            for (int i = 0; i < processor_grid_side * processor_grid_side; i++)
            {
                sending_dimension[i + 1] = full_dims[2 * i + 2] * full_dims[2 * i + 3];
            }

            for (int i = 0; i < processor_grid_side * processor_grid_side; i++)
            {

                displacements[i + 1] = displacements[i] + sending_dimension[i];
            }

            for (int i = 0; i < processor_grid_side * processor_grid_side + 1; i++)
            {
                printf("     %d     \n", sending_dimension[i]);
            }

            for (int i = 0; i < processor_grid_side * processor_grid_side + 1; i++)
            {

                printf("     %d     \n", displacements[i]);
            }
        }

        MPI_Barrier(full_group_comm);

        MPI_Gatherv(local_block, local_block_cols * local_block_rows, MPI_FLOAT, matrix_C_read, sending_dimension, displacements, MPI_FLOAT, 0, full_group_comm);

        MPI_Barrier(full_group_comm);

        for (int i = 0; i < local_block_cols * local_block_rows; i++)
        {
            printf("LOCAL BLOCK: from %d position %d:  %f\n", full_rank, i, local_block[i]);
        }

        MPI_Barrier(full_group_comm);

        if (world_rank == 0)
        {
            for (int i = 0; i < matrix_side * matrix_side; i++)
            {
                printf("MATRIX position %d %f \n", i, matrix_C_read[i]);
            }
        }
    }

    float *final_matrix = NULL;

    if (full_group_comm != MPI_COMM_NULL)
    {
        if (full_rank == 0)
        {
            final_matrix = (float *)malloc(matrix_side * matrix_side * sizeof(float)); // result of the multiplications
            // function REORDER THE MATRIX
            from_blocks_to_matrix(matrix_C_read, &final_matrix, matrix_block_structure, processor_grid_side, matrix_side);

            // produce the output csv file

            FILE *file_to_write = NULL;
            produce_csv(&file_to_write, "./data/sample_input_matrices/matrix_c.csv", final_matrix, matrix_side);
            printf("ziopersdsij");
        }
    }
    // End section, freeing

    if (full_group_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&full_group_comm);
    }
    if (cart_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&cart_comm);
    }
    if (worker_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&worker_comm);
    } // it needs to be put here since the process that got managers_and_workers equal to MPI_UNDEFINED didn't get their communicator initialized

    // free matrix of blocks
    if (block_matrix_A != NULL)
    {
        destroy_matrix_of_blocks(block_matrix_A, processor_grid_side);
    }
    if (block_matrix_B != NULL)
    {
        destroy_matrix_of_blocks(block_matrix_B, processor_grid_side);
    }

    // free pointers
    if (matrix_block_structure != NULL)
    {
        free(matrix_block_structure);
    }

    // free full matrices opened at the beginning
    if (matrix_A_read != NULL)
    {
        free(matrix_A_read);
    }
    if (matrix_B_read != NULL)
    {
        free(matrix_B_read);
    }
    if (matrix_C_read != NULL)
    {
        free(matrix_C_read);
    }

    MPI_Finalize();
    return 0;
}