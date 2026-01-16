#include "consts.h"
#include "structs.h"
#include "mpi_functions.h"
#include "mpi_profiler.h"

#include <stdio.h>
#include <time.h>

#include <sys/stat.h>
#include <errno.h>

int produce_csv(FILE **matrix_file_output, char *filename, float *matrix_final, int matrix_side)
{
    if ((*matrix_file_output = fopen(filename, "w")) == NULL)
    {
        fprintf(stderr, "Error while opening csv files.\n");
        return STATUS_READING_ERROR;
    }

    for (int row = 0; row < matrix_side; row++)
    {
        for (int col = 0; col < matrix_side; col++)
        {
            fprintf(*matrix_file_output, "%f", matrix_final[row * matrix_side + col]);
            if (col != matrix_side - 1)
            {
                fprintf(*matrix_file_output, ",");
            }
        }
        fprintf(*matrix_file_output, "\n");
    }
    return STATUS_OK;
}

int get_csv_matrix_dimension(FILE *file)
{
    char *line = NULL; // Line buffer (will be allocated by getline)
    size_t len = 0;    // Size of the buffer (getline sets it)

    int counter_lines_A = 0;
    int counter_columns_A = -1;

    while (getline(&line, &len, file) != -1)
    {
        // let's check here that the matrices have all the same dimensions
        line[strcspn(line, "\r\n")] = 0;

        int new_counter_cols = 0;

        char *token;
        token = strtok(line, ",");
        while (token != NULL)
        {
            token = strtok(NULL, ",");
            new_counter_cols++;
        }

        if (counter_columns_A != -1)
        {
            if (counter_columns_A != new_counter_cols)
            {
                rewind(file); // Good practice to reset the file state before returning.
                free(line);
                return STATUS_MATRIX_INVALID;
            }
        }
        counter_columns_A = new_counter_cols;
        counter_lines_A++;
        // perform the check
    }

    if (!(counter_columns_A == counter_lines_A))
    {
        rewind(file); // Good practice to reset the file state before returning.
        free(line);
        return STATUS_MATRIX_INVALID;
    }

    rewind(file);
    free(line);
    return counter_columns_A; // notice that the return code is 0 both if the size is zero and both if there had been an error
}

int obtain_full_matrices(const char *filename_matrix_A, const char *filename_matrix_B, MPIContext *ctx, Block ***block_matrix_A, Block ***block_matrix_B)
// to be executed only from root process
{
    FILE *file_matrix_A = NULL;
    FILE *file_matrix_B = NULL;

    if ((file_matrix_A = fopen(filename_matrix_A, "r")) == NULL)
    {
        fprintf(stderr, "Error while opening csv files.\n");
        return STATUS_READING_ERROR;
    }

    if ((file_matrix_B = fopen(filename_matrix_B, "r")) == NULL)
    {
        fprintf(stderr, "Error while opening csv files.\n");
        return STATUS_READING_ERROR;
    }

    // now detect the dimension of each matrix, ensure they are both square and have the same side
    // *matrix_A_side = get_csv_matrix_dimension(file_matrix_A);

    int matrix_A_side = get_csv_matrix_dimension(file_matrix_A);
    if (matrix_A_side == STATUS_MATRIX_INVALID)
    {
        fprintf(stderr, "Error while checking matrix size.\n");

        fclose(file_matrix_A);
        fclose(file_matrix_B);

        return STATUS_MATRIX_INVALID;
    }

    ctx->matrix_side = matrix_A_side;

    fprintf(ctx->log_file, "Matrix A dims %d\n", matrix_A_side);

    int matrix_B_side = get_csv_matrix_dimension(file_matrix_B);
    if (matrix_B_side == STATUS_MATRIX_INVALID)
    {
        fprintf(stderr, "Error while checking matrix size.\n");

        fclose(file_matrix_A);
        fclose(file_matrix_B);

        return STATUS_MATRIX_INVALID;
    }

    if ((matrix_A_side != matrix_B_side) || (matrix_A_side == 0))
    {
        fprintf(stderr, "Error while comparing matrices size.\n");

        fclose(file_matrix_A);
        fclose(file_matrix_B);

        return STATUS_MATRIX_INVALID;
    }

    float *matrix_A = NULL;
    float *matrix_B = NULL;

    // now actually read mmatrix A
    matrix_A = matrix_reader(file_matrix_A, matrix_A_side);

    // now actually read matrix B
    matrix_B = matrix_reader(file_matrix_B, matrix_B_side);
    if (matrix_A == NULL || matrix_B == NULL)
    {
        fprintf(stderr, "Error while parsing matrix.\n");

        free(matrix_A);
        free(matrix_B);

        fclose(file_matrix_A);
        fclose(file_matrix_B);

        return STATUS_MATRIX_INVALID;
    }

    if (file_matrix_A != NULL)
    {
        fclose(file_matrix_A);
    }

    if (file_matrix_B != NULL)
    {
        fclose(file_matrix_B);
    }
    // now parse the matrices

    // now divide the matrix in block and do the systolic array multiplications in blocks.
    int block_size = ctx->matrix_side / ctx->processor_grid_side;
    int block_reminder = ctx->matrix_side % ctx->processor_grid_side;
    // TODO: check for matrix bigger than grid

    fprintf(ctx->log_file, "block size %d\nblock reminder %d\n", block_size, block_reminder);

    // TODO ADD ERRORS HERE
    // analyze matrix A (and so B) structure
    int *block_struct = calculate_blocks_sizes(ctx->processor_grid_side, ctx->matrix_side);
    if (block_struct == NULL)
    {
        fprintf(stderr, "Memory error while calculating blocks.\n");

        free(matrix_A);
        free(matrix_B);

        return STATUS_ALLOCATION_FAILED;
    }

    ctx->matrix_block_structure = block_struct;

    // divide matrix A in blocks according to the processor grid.
    Block **temp_block = matrix_parser_general(matrix_A, ctx->matrix_block_structure, ctx->processor_grid_side);
    if (temp_block == NULL)
    {
        fprintf(stderr, "Memory error while parsing blocks.\n");

        free(matrix_A);
        free(matrix_B);

        return STATUS_ALLOCATION_FAILED;
    }
    *block_matrix_A = temp_block;

    // divide matrix B in blocks according to the processor grid.
    temp_block = matrix_parser_general(matrix_B, ctx->matrix_block_structure, ctx->processor_grid_side);

    if (temp_block == NULL)
    {
        fprintf(stderr, "Memory error while parsing blocks.\n");

        free(matrix_A);
        free(matrix_B);

        return STATUS_ALLOCATION_FAILED;
    }

    *block_matrix_B = temp_block;

    free(matrix_A);
    free(matrix_B);

    return STATUS_OK;
}

int open_logfiles(MPIContext *ctx, int live)
{ // to call after one has defined full_rank

    char number[5];
    int buf_expt_size = snprintf(number, sizeof(number), "%d", ctx->full_rank);
    if (buf_expt_size > sizeof(number) || buf_expt_size < 0)
    {
        fprintf(stderr, "Buffer overflow\n");
        return STATUS_BUFFER_OVERFLOW;
    }

    // safe since check before
    strcat(ctx->log_process_filepath, number);
    strcat(ctx->log_process_filepath, ctx->extension);

    if ((ctx->log_file = fopen(ctx->log_process_filepath, "w")) == NULL)
    {
        fprintf(stderr, "Error in opening log files\n");
        return STATUS_READING_ERROR;
    }

    if (live == 1)
    {
        setvbuf(ctx->log_file, NULL, _IONBF, 0); // Disable buffering completely DANGEROUS+ SLOW DOWN THE THING
    }

    return STATUS_OK;
}

void mkdir_if_missing(const char *path)
{
    if (mkdir(path, 0777) != 0 && errno != EEXIST)
    {
        perror(path);
    }
}

// this functions need to be called by all the processes
int save_info_timings(MPIContext *ctx, char *timestamp, double total_time, double distribution_computation_gathering_time)
{
    char profiler_timings_filename[128];

    // Initialize the file contaning the execution timings
    FILE *total_timings_file = NULL;
    MPI_File profiling_timings_file;

    snprintf(profiler_timings_filename, sizeof(profiler_timings_filename),
             "./data/timings/profiling_timings_%s.csv", timestamp);

    int err = 0;
    if ((err = MPI_File_open(MPI_COMM_WORLD, profiler_timings_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &profiling_timings_file)) != 0)
    {
        printf("ERROR %d\n", err);
        MPI_Abort(ctx->full_group_comm, 4512);
        return STATUS_READING_ERROR;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (ctx->full_rank == 0)
    {
        printf("Opened profiling timings file at %s\n", profiler_timings_filename);

        // Build total timings filename
        char total_timings_filename[128];

        snprintf(total_timings_filename, sizeof(total_timings_filename),
                 "./data/timings/total_timings_%s.csv", timestamp);

        if ((total_timings_file = fopen(total_timings_filename, "w")) == NULL) // TODO GIVES ERROR HERE IF FILE NOT PRESENT (chat says no) (to log a lot of files add like a hash of the time of starts here)
        {
            MPI_Abort(ctx->full_group_comm, 450);
            return STATUS_READING_ERROR;
        }
        printf("Opened total timings file at %s\n", total_timings_filename);

        const char *header_total = "TotalProcessesInvolved,MatrixDimensionSide,Total_time,Distribution_computation_gathering_time\n";

        fprintf(total_timings_file, "%s", header_total);

        fprintf(total_timings_file, "%d,%d,%.6f,%.6f\n", ctx->full_size, ctx->matrix_side, total_time, distribution_computation_gathering_time);

        fclose(total_timings_file);

        // Rank 0 writes the csv header
        const char *header_profiler = "TotalProcessesInvolved,MatrixDimensionSide,Rank,MPI_SendCount, MPI_SendTTime, MPI_RecvCount, MPI_RecvTTime,"
                                      "MPI_BcastCount, MPI_BcastTTime, MPI_SendrecvCount, MPI_SendrecvTime, MPI_ReduceCount, MPI_ReduceTTime,"
                                      "MPI_AllreduceCount, MPI_AllreduceTTime, MPI_ScatterCount, MPI_ScatterTTime, MPI_GatherCount, MPI_GatherTTime,"
                                      "MPI_GathervCount, MPI_GathervTTime\n";

        MPI_File_write_shared(profiling_timings_file, header_profiler, strlen(header_profiler), MPI_CHAR, MPI_STATUS_IGNORE);
        // so they all can see pointer advancing
    }

    MPI_Barrier(MPI_COMM_WORLD);

    char line[256];
    snprintf(line, sizeof(line), "%d,%d,%d,%.6f,%d,%.6f,%d,%.6f,%d,%.6f,%d,%.6f,%d,%.6f,%d,%.6f,%d,%.6f,%d,%.6f,%d\n",
             ctx->full_size, ctx->matrix_side, ctx->full_rank, comm_profile.send_time, comm_profile.send_count, comm_profile.recv_time, comm_profile.recv_count,
             comm_profile.bcast_time, comm_profile.bcast_count,
             comm_profile.sendrecv_time, comm_profile.sendrecv_count,
             comm_profile.reduce_time, comm_profile.reduce_count,
             comm_profile.allreduce_time, comm_profile.allreduce_count,
             comm_profile.scatter_time, comm_profile.scatter_count,
             comm_profile.gather_time, comm_profile.gather_count,
             comm_profile.gatherv_time, comm_profile.gatherv_count);

    MPI_File_write_ordered(profiling_timings_file, line, strlen(line), MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_File_close(&profiling_timings_file);
}