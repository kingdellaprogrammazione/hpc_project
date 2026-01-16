#include <time.h>
#include "mpi.h"

#ifndef MPI_PROFILER
#define MPI_PROFILER

typedef struct
{
    double send_time;
    int send_count;

    double recv_time;
    int recv_count;

    double bcast_time;
    int bcast_count;

    double sendrecv_time;
    int sendrecv_count;

    double reduce_time;
    int reduce_count;

    double allreduce_time;
    int allreduce_count;

    double scatter_time;
    int scatter_count;

    double gather_time;
    int gather_count;

    double gatherv_time;
    int gatherv_count;

    // Add more MPI calls here if needed
} MPI_CommProfiler;

extern MPI_CommProfiler comm_profile; // File-level static struct
extern int measuring;

// You can call these from your main code
void start_comm_profiling();

void stop_comm_profiling();

// ========== WRAPPERS ==========

int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest,
             int tag, MPI_Comm comm);

int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source,
             int tag, MPI_Comm comm, MPI_Status *status);
int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm);

int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag,
                 void *recvbuf, int recvcount, MPI_Datatype recvtype, int source, int recvtag,
                 MPI_Comm comm, MPI_Status *status);

int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm);

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);

int MPI_Scatter(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, int recvcount, MPI_Datatype recvtype,
                int root, MPI_Comm comm);
int MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
               void *recvbuf, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm comm);

int MPI_Gatherv(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                void *recvbuf, const int *recvcounts, const int *displs,
                MPI_Datatype recvtype, int root, MPI_Comm comm);

void print_comm_profile(int rank);
#endif
