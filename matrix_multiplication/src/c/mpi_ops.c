#include "mpi.h"
#include "structs.h"
#include <stdlib.h>
#include <assert.h>
#include "mpi_functions.h"
#include "consts.h"

void organize_processors(MPIContext *ctx)
{
    int color = 1;

    if (ctx->processor_grid_side * ctx->processor_grid_side != ctx->world_size)
    {
        if ((ctx->world_rank > 0) && (ctx->world_rank < (ctx->processor_grid_side * ctx->processor_grid_side + 1)))
        {
            color = 1;
        }
        else
        {
            color = MPI_UNDEFINED;
        }
    }

    MPI_Comm_split(MPI_COMM_WORLD, color, ctx->world_rank, &(ctx->worker_comm)); // edge case of square world size handled

    if (ctx->worker_comm != MPI_COMM_NULL) // consider only workers
    {
        // Create the Cartesian grid from the new, smaller communicator
        const int ndims = 2;
        int dims[2] = {ctx->processor_grid_side, ctx->processor_grid_side};
        int periods[2] = {0, 0};
        int reorder = 1;

        MPI_Cart_create(ctx->worker_comm, ndims, dims, periods, reorder, &(ctx->cart_comm));

        // --- Now, use the grid communicator for your work ---
        MPI_Comm_rank(ctx->cart_comm, &(ctx->cart_rank));
        MPI_Cart_coords(ctx->cart_comm, ctx->cart_rank, ndims, ctx->cartesian_coords);

        // fprintf(ctx->log_file, "Full Rank %2d: ACTIVE -> Grid Rank %2d -> Coords (%d, %d)\n",
        //         ctx->full_rank, ctx->cart_rank, ctx->cartesian_coords[0], ctx->cartesian_coords[1]);
    }

    if (ctx->world_rank == 0)
    {
        color = 1;
    }

    MPI_Comm_split(MPI_COMM_WORLD, color, ctx->world_rank, &(ctx->full_group_comm)); // this split includes all processes, also the manager, should work also in the edge case

    if (ctx->full_group_comm != MPI_COMM_NULL)
    {
        MPI_Comm_size(ctx->full_group_comm, &(ctx->full_size));
        MPI_Comm_rank(ctx->full_group_comm, &(ctx->full_rank));
    }
    if (ctx->cart_comm != MPI_COMM_NULL)
    {
        MPI_Comm_size(ctx->cart_comm, &(ctx->cart_size));
        MPI_Comm_rank(ctx->cart_comm, &(ctx->cart_rank));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

void obtain_rank_conversion(MPIContext *ctx, int **full_comm_ranks, int **rank_lookup_table_cartesian)
{

    *full_comm_ranks = (int *)malloc(ctx->cart_size * sizeof(int));

    int cart_rank_in_full_world = -1;
    // We use an Allreduce to find the rank of the process where cart_rank is 0.
    // The process with cart_rank == 0 contributes its full_rank. Everyone else contributes -1.
    // The MPI_MAX operation will find the one valid rank.
    int root_candidate = (ctx->cart_rank == 0) ? ctx->full_rank : -1; // the case for cart_rank  == -1 is automatically escluded

    MPI_Allreduce(&root_candidate, &cart_rank_in_full_world, 1, MPI_INT, MPI_MAX, ctx->full_group_comm);

    if (ctx->cart_rank == 0) // this needs to be executed from a process that belongs to both groups
    {

        // STEP 2: Translate this rank into even_comm
        MPI_Group cart_group, full_group;
        MPI_Comm_group(ctx->cart_comm, &cart_group);
        MPI_Comm_group(ctx->full_group_comm, &full_group);

        int rank_to_translate[ctx->cart_size];
        for (int i = 0; i < ctx->cart_size; i++)
        {
            rank_to_translate[i] = i;
        }

        MPI_Group_translate_ranks(cart_group, ctx->cart_size, rank_to_translate, full_group, *full_comm_ranks);
        for (int i = 0; i < ctx->cart_size; i++)
        {
            fprintf(ctx->log_file, "Rank %d in cartesian is rank %d in bigger group\n", i, (*full_comm_ranks)[i]);
        }
        // Clean up groups
        MPI_Group_free(&cart_group);
        MPI_Group_free(&full_group);

        MPI_Bcast(*full_comm_ranks, ctx->cart_size, MPI_INT, cart_rank_in_full_world, ctx->full_group_comm);

        *rank_lookup_table_cartesian = malloc(ctx->processor_grid_side * ctx->processor_grid_side * sizeof(int));
        for (int i = 0; i < ctx->processor_grid_side; i++)
        {
            for (int j = 0; j < ctx->processor_grid_side; j++)
            {
                int temp_coords[2] = {i, j};
                int temp_cart_rank;
                // This call is legal here because this process is in cart_comm!
                MPI_Cart_rank(ctx->cart_comm, temp_coords, &temp_cart_rank);
                // Use the translated ranks to find the final destination rank.
                (*rank_lookup_table_cartesian)[i * ctx->processor_grid_side + j] = (*full_comm_ranks)[temp_cart_rank];
            }
        }
        MPI_Send(*rank_lookup_table_cartesian, ctx->processor_grid_side * ctx->processor_grid_side, MPI_INT, 0, 99, ctx->full_group_comm);
    }
    if (ctx->full_rank == 0)
    {
        *rank_lookup_table_cartesian = malloc(ctx->processor_grid_side * ctx->processor_grid_side * sizeof(int));
        // If the manager IS the root worker (e.g., 1 proc), it doesn't need to Recv.

        MPI_Recv(*rank_lookup_table_cartesian, ctx->processor_grid_side * ctx->processor_grid_side, MPI_INT, cart_rank_in_full_world, 99, ctx->full_group_comm, MPI_STATUS_IGNORE);

        fprintf(ctx->log_file, "Manager has received the rank lookup table.\n");
    }
}

void printonrank(char *string, int rank, MPI_Comm comm)
{
    if (comm != MPI_COMM_NULL)
    {
        int process_rank;
        MPI_Comm_rank(comm, &process_rank);
        if (process_rank == rank)
        {
            printf("%s", string);
        }
    }
}

void cartesian_explorer(MPIContext *ctx)
{

    fprintf(ctx->log_file, "Cartesian Rank %d Rows %d Cols %d \n", ctx->cart_rank, ctx->local_block_rows, ctx->local_block_cols);
    int block_total_elements = ctx->local_block_rows * ctx->local_block_cols; // individual result for each block

    ctx->multi_result = (float *)malloc(block_total_elements * sizeof(float)); // result of the multiplications
    ctx->local_block = (float *)malloc(block_total_elements * sizeof(float));  // locally hosted block

    assert(block_total_elements != 0);
    if (ctx->multi_result == NULL || ctx->local_block == NULL)
    {
        printf("P%d: Failed to allocate memory for local blocks.\n", ctx->full_rank);
        MPI_Abort(MPI_COMM_WORLD, 1); // A hard stop for errors
    }

    // Discover vertical neighbors
    MPI_Cart_shift(ctx->cart_comm, 0, 1, &(ctx->receivefrom_vertical), &(ctx->sendto_vertical));

    // Discover horizontal neighbors
    MPI_Cart_shift(ctx->cart_comm, 1, 1, &(ctx->receivefrom_horizontal), &(ctx->sendto_horizontal));

    ctx->multi_result = create_zero_matrix(block_total_elements);
    ctx->local_block = create_zero_matrix(block_total_elements);
    fprintf(ctx->log_file, "Allocated memory successfully for single grid processor\n");
    MPI_Barrier(ctx->cart_comm);
}

int scatter_block_dims(MPIContext *ctx, Block **block_matrix_A)
{
    int *array_rows = NULL;
    int *array_cols = NULL;

    if (ctx->full_rank == 0)
    {
        int total_blocks = ctx->processor_grid_side * ctx->processor_grid_side;
        array_rows = malloc((total_blocks + 1) * sizeof(int));
        array_cols = malloc((total_blocks + 1) * sizeof(int));
        if (!array_rows || !array_cols)
        {
            fprintf(stderr, "Failed to allocate memory for scatter arrays.\n");
            free(array_rows);
            free(array_cols);
            return STATUS_ALLOCATION_FAILED;
        }

        array_rows[0] = 0;
        array_cols[0] = 0;

        fprintf(ctx->log_file, "Rows in block 0 are %d\n", array_rows[0]);
        fprintf(ctx->log_file, "Cols in block 0 are %d\n", array_cols[0]);

        for (int i = 0; i < total_blocks; i++)
        {
            array_rows[i + 1] = block_matrix_A[i]->dims[0];
            array_cols[i + 1] = block_matrix_A[i]->dims[1];
            fprintf(ctx->log_file, "Rows in block %d are %d\n", i + 1, array_rows[i + 1]);
            fprintf(ctx->log_file, "Cols in block %d are %d\n", i + 1, array_cols[i + 1]);
        }
    }

    MPI_Scatter(array_rows, 1, MPI_INT, &(ctx->local_block_rows), 1, MPI_INT, 0, ctx->full_group_comm);
    MPI_Scatter(array_cols, 1, MPI_INT, &(ctx->local_block_cols), 1, MPI_INT, 0, ctx->full_group_comm);

    if (ctx->full_rank == 0)
    {
        free(array_rows);
        free(array_cols);
    }

    return STATUS_OK;
}

void send_dimensions_matrix_left(MPIContext *ctx, int *rank_lookup_table, Block **block_matrix, int start_index, int end_index, int clock)
{
    // send on the left side matrix A
    for (int i = 0; i < ctx->processor_grid_side; i++)
    {
        int rank_to_send_to = rank_lookup_table[i * ctx->processor_grid_side + 0];
        int null_dims[2] = {0, 0};

        if (i >= start_index && i < end_index)
        {
            int index_row = i;
            int index_col = clock - i;
            // go from coordinates to cartesian grid rank , I do not need this in reality, only for code clarity

            MPI_Send(block_matrix[index_row * ctx->processor_grid_side + index_col]->dims, 2, MPI_INT, rank_to_send_to, 0, ctx->full_group_comm); // send info about the incoming block's dimensions

            fprintf(ctx->log_file, "Clock %d Process %d, full_rank %d, cart_rank %d MASTER sending the dims to proc full_rank %d\n", clock, ctx->full_rank, ctx->full_rank, ctx->cart_rank, rank_to_send_to);
        }
        else
        {
            MPI_Send(null_dims, 2, MPI_INT, rank_to_send_to, 0, ctx->full_group_comm); // send info about the incoming block's dimensions
                                                                                       //  MPI_Send(NULL, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                                                                                       // mandare blocco nullo  // non mando nulla
            fprintf(ctx->log_file, "Clock %d Process %d, full_rank %d, cart_rank %d MASTER sending the NULL DIMS to proc full_rank %d\n", clock, ctx->full_rank, ctx->full_rank, ctx->cart_rank, rank_to_send_to);
        }
    }
}

void send_dimensions_matrix_top(MPIContext *ctx, int *rank_lookup_table, Block **block_matrix, int start_index, int end_index, int clock)
{
    for (int i = 0; i < ctx->processor_grid_side; i++)
    {
        int rank_to_send_to = rank_lookup_table[0 * ctx->processor_grid_side + i];
        int null_dims[2] = {0, 0};

        if (i >= start_index && i < end_index)
        {
            int index_row = clock - i;
            int index_col = i;

            MPI_Send(block_matrix[index_row * ctx->processor_grid_side + index_col]->dims, 2, MPI_INT, rank_to_send_to, 0, ctx->full_group_comm); // send info about the incoming block's dimensions
            fprintf(ctx->log_file, "Clock %d, Process %d, full_rank %d, cart_rank %d MASTER sending the dims to proc full_rank %d\n", clock, ctx->full_rank, ctx->full_rank, ctx->cart_rank, rank_to_send_to);
        }
        else
        {
            MPI_Send(null_dims, 2, MPI_INT, rank_to_send_to, 0, ctx->full_group_comm); // send info about the incoming block's dimensions
                                                                                       //  MPI_Send(NULL, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
                                                                                       // mandare blocco nullo  // non mando nulla
            fprintf(ctx->log_file, "Clock %d, Process %d, full_rank %d, cart_rank %d MASTER sending the NULL DIMS to proc full_rank %d\n", clock, ctx->full_rank, ctx->full_rank, ctx->cart_rank, rank_to_send_to);
        }
    }
}

void receive_exchange_dims_grid(MPIContext *ctx, int clock)
{

    int block_total_elements = ctx->local_block_rows * ctx->local_block_cols; // local

    if ((ctx->cartesian_coords)[1] == 0 && (ctx->cartesian_coords)[0] == 0) // on the corner
    {

        // first check to have received a non zero block
        MPI_Recv(ctx->coming_block_dims_horizontal, 2, MPI_INT, 0, 0, ctx->full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator

        MPI_Recv(ctx->coming_block_dims_vertical, 2, MPI_INT, 0, 0, ctx->full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator

        fprintf(ctx->log_file, "Clock %d Process %d WORKER cart_rank %d receiving the DIMS from proc%d\n", clock, ctx->full_rank, ctx->cart_rank, 0);
    }

    // remember everyone receives from the 0 rank process
    else if ((ctx->cartesian_coords)[1] == 0 && (ctx->cartesian_coords)[0] != 0) // on the left border
    {
        // first check to have received a non zero block

        MPI_Recv(ctx->coming_block_dims_horizontal, 2, MPI_INT, 0, 0, ctx->full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
        fprintf(ctx->log_file, "Clock %d Process %d WORKER cart_rank %d receiving the horizontal DIMS %d %d from proc full_rank %d\n", clock, ctx->full_rank, ctx->cart_rank, ctx->coming_block_dims_horizontal[0], ctx->coming_block_dims_horizontal[1], 0);
    }
    else if ((ctx->cartesian_coords)[0] == 0 && (ctx->cartesian_coords)[1] != 0) // on the top border
    {
        MPI_Recv(ctx->coming_block_dims_vertical, 2, MPI_INT, 0, 0, ctx->full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
        fprintf(ctx->log_file, "Clock %d Process %d WORKER cart_rank %d receiving the vertical DIMS %d %d from proc full_rank %d\n", clock, ctx->full_rank, ctx->cart_rank, ctx->coming_block_dims_vertical[0], ctx->coming_block_dims_vertical[1], 0);

        // Send the block
    }

    // devono eseguire tutti il blocco seguente, non

    // everyone needs to execute this
    MPI_Sendrecv(                                                                  // no op if the source and objective are MPI_NULL_PROC
        ctx->going_block_dims_vertical, 2, MPI_INT, ctx->sendto_vertical, 0,       // Send my data to the right
        ctx->coming_block_dims_vertical, 2, MPI_INT, ctx->receivefrom_vertical, 0, // Receive new data from the left
        ctx->cart_comm, MPI_STATUS_IGNORE);
    fprintf(ctx->log_file, "Clock %d, Process %d WORKER cart_rank %d send vertical DIMS %d %d to proc full_rank %d and receive from%d\n", clock, ctx->full_rank, ctx->cart_rank, ctx->sendto_vertical, ctx->receivefrom_vertical);

    MPI_Sendrecv(
        ctx->going_block_dims_horizontal, 2, MPI_INT, ctx->sendto_horizontal, 0,       // Send my data to the right
        ctx->coming_block_dims_horizontal, 2, MPI_INT, ctx->receivefrom_horizontal, 0, // Receive new data from the left
        ctx->cart_comm, MPI_STATUS_IGNORE);
    fprintf(ctx->log_file, "Clock %d,Process %d WORKER cart_rank %d send horizontal DIMS %d %d to proc%d and receive from%d\n", clock, ctx->full_rank, ctx->cart_rank, ctx->sendto_horizontal, ctx->receivefrom_horizontal);
}

void send_blocks_matrix_left(MPIContext *ctx, int *rank_lookup_table, Block **block_matrix, int start_index, int end_index, int clock)
{
    for (int i = 0; i < ctx->processor_grid_side; i++)
    {
        int rank_to_send_to = rank_lookup_table[i * ctx->processor_grid_side + 0];

        if (i >= start_index && i < end_index)
        {
            int index_row = i;
            int index_col = clock - i;
            // go from coordinates to cartesian grid rank , I do not need this in reality, only for code clarity

            int block_total_elements = block_matrix[index_row * ctx->processor_grid_side + index_col]->dims[0] * block_matrix[index_row * ctx->processor_grid_side + index_col]->dims[1];
            fprintf(ctx->log_file, "CHECK Process %d MASTER %d\n", ctx->full_rank, block_total_elements);

            MPI_Send(block_matrix[index_row * ctx->processor_grid_side + index_col]->ptr_to_block_contents, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, ctx->full_group_comm);
            fprintf(ctx->log_file, "Process %d MASTER sending the BLOCK to proc%d\n", ctx->full_rank, rank_to_send_to);
        }
        else
        {
            //  MPI_Send(NULL, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
            // mandare blocco nullo  // non mando nulla
            MPI_Send(NULL, 0, MPI_FLOAT, rank_to_send_to, 0, ctx->full_group_comm);
            fprintf(ctx->log_file, "Process %d MASTER sending the NULL BLOCK to proc%d\n", ctx->full_rank, rank_to_send_to);
        }
    }
}

void send_blocks_matrix_top(MPIContext *ctx, int *rank_lookup_table, Block **block_matrix, int start_index, int end_index, int clock)
{
    // send on the top side matrix B
    for (int i = 0; i < ctx->processor_grid_side; i++)
    {
        int rank_to_send_to = rank_lookup_table[0 * ctx->processor_grid_side + i];

        if (i >= start_index && i < end_index)
        {
            int index_row = clock - i;
            int index_col = i;

            int block_total_elements = block_matrix[index_row * ctx->processor_grid_side + index_col]->dims[0] * block_matrix[index_row * ctx->processor_grid_side + index_col]->dims[1];

            MPI_Send(block_matrix[index_row * ctx->processor_grid_side + index_col]->ptr_to_block_contents, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, ctx->full_group_comm);
            fprintf(ctx->log_file, "Process %d MASTER sending the BLOCK to proc%d\n", ctx->full_rank, rank_to_send_to);
        }
        else
        {
            //  MPI_Send(NULL, block_total_elements, MPI_FLOAT, rank_to_send_to, 0, full_group_comm);
            // mandare blocco nullo  // non mando nulla
            MPI_Send(NULL, 0, MPI_FLOAT, rank_to_send_to, 0, ctx->full_group_comm);
            fprintf(ctx->log_file, "Process %d MASTER sending the NULL BLOCK to proc%d\n", ctx->full_rank, rank_to_send_to);
        }
    }
}

void receive_exchange_blocks_grid(MPIContext *ctx, int clock)
{
    int coming_horizontal_blocks_elements = ctx->coming_block_dims_horizontal[0] * ctx->coming_block_dims_horizontal[1];
    int coming_vertical_blocks_elements = ctx->coming_block_dims_vertical[0] * ctx->coming_block_dims_vertical[1]; // TODO CHECK IF WE CAN SIMPLIFY LOGIC ASSUMING THE DIM IS SAME WITH HORIZONTAL

    fprintf(ctx->log_file, "cart_rank %d Iteration %d, coming block horizontal dim %d , coming block vertical %d\n", ctx->cart_rank, clock, coming_horizontal_blocks_elements, coming_vertical_blocks_elements);
    // i sent the dimensions of the blocks coming and going // aggiungere invii di blocchi non dovrebbe creare problematiche in quanto si sta svolgendo tutto in parallelo.

    if (ctx->coming_block_dims_horizontal[0] * ctx->coming_block_dims_horizontal[1] != 0)
    {
        ctx->coming_block_horizontal = (float *)malloc(ctx->coming_block_dims_horizontal[0] * ctx->coming_block_dims_horizontal[1] * sizeof(float));
    }
    else
    {
        ctx->coming_block_horizontal = NULL;
    }
    if (ctx->coming_block_dims_vertical[0] * ctx->coming_block_dims_vertical[1] != 0)
    {
        ctx->coming_block_vertical = (float *)malloc(ctx->coming_block_dims_vertical[0] * ctx->coming_block_dims_vertical[1] * sizeof(float));
    }
    else
    {
        ctx->coming_block_vertical = NULL;
    }

    // if (clock == 0)
    // {
    //     ctx->going_block_horizontal = (float *)malloc(ctx->going_block_dims_horizontal[0] * ctx->going_block_dims_horizontal[1] * sizeof(float));
    //     ctx->going_block_vertical = (float *)malloc(ctx->going_block_dims_vertical[0] * ctx->going_block_dims_vertical[1] * sizeof(float));
    // }

    if (ctx->cartesian_coords[1] == 0 && ctx->cartesian_coords[0] == 0) // on the corner
    {
        MPI_Recv(ctx->coming_block_horizontal, coming_horizontal_blocks_elements, MPI_FLOAT, 0, 0, ctx->full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
        MPI_Recv(ctx->coming_block_vertical, coming_vertical_blocks_elements, MPI_FLOAT, 0, 0, ctx->full_group_comm, MPI_STATUS_IGNORE);     // remember here the right large communicator

        fprintf(ctx->log_file, "Process %d WORKER receiving the BLOCK from proc%d, %d\n", ctx->full_rank, 0, 0);
    }

    // remember everyone receives from the 0 rank process
    else if (ctx->cartesian_coords[1] == 0 && ctx->cartesian_coords[0] != 0) // on the left border
    {
        MPI_Recv(ctx->coming_block_horizontal, coming_horizontal_blocks_elements, MPI_FLOAT, 0, 0, ctx->full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
        fprintf(ctx->log_file, "Process %d WORKER receiving the BLOCK from proc%d\n", ctx->full_rank, 0);
    }
    else if (ctx->cartesian_coords[0] == 0 && ctx->cartesian_coords[1] != 0) // on the top border
    {
        MPI_Recv(ctx->coming_block_vertical, coming_vertical_blocks_elements, MPI_FLOAT, 0, 0, ctx->full_group_comm, MPI_STATUS_IGNORE); // remember here the right large communicator
        fprintf(ctx->log_file, "Process %d WORKER receiving the BLOCK from proc%d\n", ctx->full_rank, 0);
    }

    // send current blocks

    MPI_Sendrecv(
        ctx->going_block_horizontal, (ctx->going_block_dims_horizontal)[0] * (ctx->going_block_dims_horizontal)[1], MPI_FLOAT, ctx->sendto_horizontal, 0,         // Send my data to the right
        ctx->coming_block_horizontal, (ctx->coming_block_dims_horizontal)[0] * (ctx->coming_block_dims_horizontal)[1], MPI_FLOAT, ctx->receivefrom_horizontal, 0, // Receive new data from the left
        ctx->cart_comm, MPI_STATUS_IGNORE);

    fprintf(ctx->log_file, "Process %d WORKER send BLOCKS from proc%d and receive from%d\n", ctx->full_rank, ctx->sendto_horizontal, ctx->receivefrom_horizontal);

    MPI_Sendrecv(
        ctx->going_block_vertical, (ctx->going_block_dims_vertical)[0] * (ctx->going_block_dims_vertical)[1], MPI_FLOAT, ctx->sendto_vertical, 0,         // Send my data to the right
        ctx->coming_block_vertical, (ctx->coming_block_dims_vertical)[0] * (ctx->coming_block_dims_vertical)[1], MPI_FLOAT, ctx->receivefrom_vertical, 0, // Receive new data from the left
        ctx->cart_comm, MPI_STATUS_IGNORE);

    fprintf(ctx->log_file, "Process %d WORKER send BLOCKS from proc%d and receive from%d\n", ctx->full_rank, ctx->sendto_vertical, ctx->receivefrom_vertical);
}

void run_local_calculation(MPIContext *ctx, int clock)
{
    matrix_multi(ctx->coming_block_horizontal, ctx->coming_block_vertical, ctx->multi_result, ctx->coming_block_dims_horizontal[0], ctx->coming_block_dims_horizontal[1], ctx->coming_block_dims_vertical[1]);

    if (ctx->cart_rank == 0)
    {
        fprintf(ctx->log_file, "Clock %d +-+-+-+--+ \n", clock);
        show_matrix(ctx->coming_block_horizontal, ctx->coming_block_dims_horizontal[0], ctx->coming_block_dims_horizontal[1]);
        fprintf(ctx->log_file, "Clock %d \n", rand());

        show_matrix(ctx->coming_block_vertical, ctx->coming_block_dims_vertical[0], ctx->coming_block_dims_vertical[1]);
        fprintf(ctx->log_file, "Clock %d \n", rand());

        show_matrix(ctx->multi_result, ctx->coming_block_dims_horizontal[0], ctx->coming_block_dims_vertical[1]);
    }
    // test
    float *temp = NULL;
    temp = (float *)malloc(ctx->local_block_cols * ctx->local_block_rows * sizeof(float));

    matrix_add(ctx->multi_result, ctx->local_block, temp, ctx->local_block_rows, ctx->local_block_cols); // attention here TODO check
    ctx->local_block = temp;
    if (ctx->cart_rank == 0)
    {
        fprintf(ctx->log_file, "--------------------------\n");
        show_matrix(ctx->local_block, ctx->local_block_rows, ctx->local_block_cols);
        fprintf(ctx->log_file, "--------------------------\n");
    }
}

void prepare_next_clock(MPIContext *ctx)
{
    // update blocks and dims

    ctx->going_block_dims_horizontal[0] = ctx->coming_block_dims_horizontal[0];
    ctx->going_block_dims_vertical[0] = ctx->coming_block_dims_vertical[0];
    ctx->going_block_dims_horizontal[1] = ctx->coming_block_dims_horizontal[1];
    ctx->going_block_dims_vertical[1] = ctx->coming_block_dims_vertical[1];

    if (ctx->going_block_horizontal != NULL)
    {
        free(ctx->going_block_horizontal);
    }
    if (ctx->going_block_vertical != NULL)
    {
        free(ctx->going_block_vertical);
    }
    ctx->going_block_vertical = ctx->coming_block_vertical;
    ctx->going_block_horizontal = ctx->coming_block_horizontal;

    ctx->coming_block_vertical = NULL;
    ctx->coming_block_horizontal = NULL;

    if (ctx->multi_result != NULL)
    {
        free(ctx->multi_result);
        ctx->multi_result = create_zero_matrix(ctx->local_block_cols * ctx->local_block_rows);
    }
}

void main_loop(MPIContext *ctx, int *rank_lookup_table, Block **block_matrix_A, Block **block_matrix_B)
{
    // set up pulses
    int pulses = ctx->processor_grid_side * 3 - 1; // evaluate number of systolic pulses , consider also the pulses to empty the array
    // TODO RISPARMIARE TEMPO QUANDO SO CHE STO FACENDO MOLTIPLICAZIONI CON MATRICI A ZERO,
    // comunque non perdiamo tempo perche ci sono alcune moltiplicazioni che effettivamente vengono effettuate

    for (int clock = 0; clock < pulses; clock++)
    {
        fprintf(ctx->log_file, "Process full_rank %d, cart_rank %d: entering clock %d.\n", ctx->full_rank, ctx->cart_rank, clock);
        // prepare the first blocks to send

        int start_index = 0; // incluso
        int end_index = 1;   // escluso

        if (ctx->full_rank == 0)
        {

            start_index = clock - ctx->processor_grid_side + 1; // controllare questo TODO
            if (start_index < 0)
            {
                start_index = 0;
            }

            end_index = clock + 1;
            if (end_index > ctx->processor_grid_side)
            {
                end_index = ctx->processor_grid_side; // escluso
            }

            int null_dims[2] = {0, 0};

            fprintf(ctx->log_file, "Clock %d, Indexes edges %d %d.\n", clock, start_index, end_index);

            // send on the left side matrix A
            send_dimensions_matrix_left(ctx, rank_lookup_table, block_matrix_A, start_index, end_index, clock);

            // TODO ADD A SORT OF STATUS STRUCT WITH THE CLOCK AND THE INDEX

            // send on the top side matrix B
            send_dimensions_matrix_top(ctx, rank_lookup_table, block_matrix_B, start_index, end_index, clock);
        }

        MPI_Barrier(ctx->full_group_comm);
        fprintf(ctx->log_file, "Clock %d Distribution from MASTER ended. Starting the grid's work.\n", clock);

        // considering only worker grid process
        if (ctx->cart_comm != MPI_COMM_NULL) // remember that until explicitly changed the incoming and outcoming dimensions are 0.
        {
            receive_exchange_dims_grid(ctx, clock);
        }

        MPI_Barrier(ctx->full_group_comm);
        fprintf(ctx->log_file, "Clock %d Ended dimension distribution. Block distribution starting.\n", clock);

        // we've sent the dimensions, now send the blocks

        if (ctx->full_rank == 0)
        {

            send_blocks_matrix_left(ctx, rank_lookup_table, block_matrix_A, start_index, end_index, clock);

            send_blocks_matrix_top(ctx, rank_lookup_table, block_matrix_B, start_index, end_index, clock);
        }

        MPI_Barrier(ctx->full_group_comm);
        fprintf(ctx->log_file, "Ci siamo\n");

        if (ctx->cart_comm != MPI_COMM_NULL)
        {
            receive_exchange_blocks_grid(ctx, clock);
            // CALCULATE

            // Compute
            // TODO CHECK COMMON DIMENSION OF INCOMING BLOCKS

            MPI_Barrier(ctx->cart_comm);

            if (ctx->coming_block_dims_horizontal != NULL && ctx->coming_block_dims_horizontal != NULL) // the zeros arrive simultaneously
            {
                run_local_calculation(ctx, clock);
            }
            // create zero blocks per sicurezza

            prepare_next_clock(ctx);
        }

        // /// debuggggg
        //  if (ctx->full_group_comm != MPI_COMM_NULL)
        // {
        //      log_MPIContext(&local_setup);
        //
        //     fclose(ctx->log_file);
        // }
        //
        //  MPI_Barrier(MPI_COMM_WORLD);
        //  MPI_Abort(MPI_COMM_WORLD, 123);
        // /// debuggggg
    }

    fprintf(ctx->log_file, "Ziopers\n");
    // CALCULATE
    MPI_Barrier(ctx->full_group_comm);
}
