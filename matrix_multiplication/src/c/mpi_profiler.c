#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include "mpi_profiler.h"
#include <string.h>
#include <time.h>

MPI_CommProfiler comm_profile = {0}; // File-level static struct
int measuring = 0;

// You can call these from your main code
void start_comm_profiling()
{
    measuring = 1;
    // Reset all timers and counts to zero
    memset(&comm_profile, 0, sizeof(comm_profile));
}

void stop_comm_profiling()
{
    measuring = 0;
}

// ========== WRAPPERS ==========

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm)
{
    double start, end;
    if (measuring)
        start = MPI_Wtime();

    int ret = PMPI_Send(buf, count, datatype, dest, tag, comm);

    if (measuring)
    {
        end = MPI_Wtime();
        comm_profile.send_time += (end - start);
        comm_profile.send_count++;
    }
    return ret;
}

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
             int tag, MPI_Comm comm, MPI_Status *status)
{
    double start, end;
    if (measuring)
        start = MPI_Wtime();

    int ret = PMPI_Recv(buf, count, datatype, source, tag, comm, status);

    if (measuring)
    {
        end = MPI_Wtime();
        comm_profile.recv_time += (end - start);
        comm_profile.recv_count++;
    }
    return ret;
}

int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm)
{
    double start, end;
    if (measuring)
        start = MPI_Wtime();

    int ret = PMPI_Bcast(buffer, count, datatype, root, comm);

    if (measuring)
    {
        end = MPI_Wtime();
        comm_profile.bcast_time += (end - start);
        comm_profile.bcast_count++;
    }
    return ret;
}

int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status)
{
    double start, end;
    if (measuring)
        start = MPI_Wtime();

    int ret = PMPI_Sendrecv(sendbuf, sendcount, sendtype, dest, sendtag,
                            recvbuf, recvcount, recvtype, source, recvtag,
                            comm, status);

    if (measuring)
    {
        end = MPI_Wtime();
        comm_profile.sendrecv_time += (end - start);
        comm_profile.sendrecv_count++;
    }
    return ret;
}

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    double start, end;
    if (measuring)
        start = MPI_Wtime();

    int ret = PMPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);

    if (measuring)
    {
        end = MPI_Wtime();
        comm_profile.reduce_time += (end - start);
        comm_profile.reduce_count++;
    }
    return ret;
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
{
    double start, end;
    if (measuring)
        start = MPI_Wtime();

    int ret = PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);

    if (measuring)
    {
        end = MPI_Wtime();
        comm_profile.allreduce_time += (end - start);
        comm_profile.allreduce_count++;
    }
    return ret;
}

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm)
{
    double start, end;
    if (measuring)
        start = MPI_Wtime();

    int ret = PMPI_Scatter(sendbuf, sendcount, sendtype,
                           recvbuf, recvcount, recvtype,
                           root, comm);

    if (measuring)
    {
        end = MPI_Wtime();
        comm_profile.scatter_time += (end - start);
        comm_profile.scatter_count++;
    }
    return ret;
}

int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm)
{
    double start, end;
    if (measuring)
        start = MPI_Wtime();

    int ret = PMPI_Gather(sendbuf, sendcount, sendtype,
                          recvbuf, recvcount, recvtype,
                          root, comm);

    if (measuring)
    {
        end = MPI_Wtime();
        comm_profile.gather_time += (end - start);
        comm_profile.gather_count++;
    }
    return ret;
}

int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, const int *recvcounts, const int *displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm)
{
    double start, end;
    if (measuring)
        start = MPI_Wtime();

    int ret = PMPI_Gatherv(sendbuf, sendcount, sendtype,
                           recvbuf, recvcounts, displs, recvtype,
                           root, comm);

    if (measuring)
    {
        end = MPI_Wtime();
        comm_profile.gatherv_time += (end - start);
        comm_profile.gatherv_count++;
    }
    return ret;
}

void print_comm_profile(int rank)
{
    printf("Rank %d Communication Timings (seconds):\n", rank);
    printf("  MPI_Send:       %.6f seconds, %d calls\n", comm_profile.send_time, comm_profile.send_count);
    printf("  MPI_Recv:       %.6f seconds, %d calls\n", comm_profile.recv_time, comm_profile.recv_count);
    printf("  MPI_Bcast:      %.6f seconds, %d calls\n", comm_profile.bcast_time, comm_profile.bcast_count);
    printf("  MPI_Sendrecv:   %.6f seconds, %d calls\n", comm_profile.sendrecv_time, comm_profile.sendrecv_count);
    printf("  MPI_Reduce:     %.6f seconds, %d calls\n", comm_profile.reduce_time, comm_profile.reduce_count);
    printf("  MPI_Allreduce:  %.6f seconds, %d calls\n", comm_profile.allreduce_time, comm_profile.allreduce_count);
    printf("  MPI_Scatter:    %.6f seconds, %d calls\n", comm_profile.scatter_time, comm_profile.scatter_count);
    printf("  MPI_Gather:     %.6f seconds, %d calls\n", comm_profile.gather_time, comm_profile.gather_count);
    printf("  MPI_Gatherv:    %.6f seconds, %d calls\n", comm_profile.gatherv_time, comm_profile.gatherv_count);
}
