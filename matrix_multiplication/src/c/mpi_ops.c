#include "mpi.h"
#include "structs.h"
#include <stdlib.h>
#include <assert.h>
#include "mpi_functions.h"

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
    if (ctx->cart_comm != MPI_COMM_NULL) // here we initialize and allocate memory only on the cartesian grid
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
}
