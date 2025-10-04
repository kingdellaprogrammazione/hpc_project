// heat_diffusion_omp.c
// Simulation of heat diffusion on a 1024x1024 grid using OpenMP
// Case (a): average of the 4 neighboring cells
// Case (b): anisotropic diffusion with wx=0.3, wy=0.2
// Initialization:
//   (a) left 50% of the plate at 250°C, right 50% at Earth's average temperature (default 14°C)
//   (b) left 25% of the plate at 540°C, right 75% at Earth's average temperature
// Boundary conditions: fixed (Dirichlet) — border cells are not updated and remain at their initial value.
// Convergence: iterates until max_iter is reached or max_delta < tol (if tol>0).
// Sampling: saves CSV files with metrics and (optionally) grid snapshots.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#ifndef N_DEFAULT
#define N_DEFAULT 1024
#endif

typedef enum { SCENARIO_A = 0, SCENARIO_B = 1 } Scenario;

typedef struct {
    int N;                  // grid size
    int max_iter;           // maximum number of iterations
    double tol;             // convergence threshold (<=0 disables it)
    Scenario scenario;      // 0 = (a), 1 = (b)
    double T_earth;         // Earth's average temperature (default 14.0)
    double wx;              // weights for scenario b (default 0.3)
    double wy;              // weights for scenario b (default 0.2)
    int threads;            // number of OpenMP threads (<=0: use OMP_NUM_THREADS)
    int sample_every;       // sampling period (iterations) for logs and snapshots
    int dump_snapshots;     // 1 = save grid snapshots as CSV
    const char* metrics_csv;// output file for metrics
    const char* snapshot_prefix; // prefix for snapshot filenames
} Params;

static inline size_t idx(int i, int j, int N){ return (size_t)i*(size_t)N + (size_t)j; }

void init_grid(double* grid, const Params* p){
    int N = p->N;
    // Default: entire grid initialized to T_earth
    #pragma omp parallel for schedule(static)
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            grid[idx(i,j,N)] = p->T_earth;
        }
    }
    if(p->scenario == SCENARIO_A){
        // left 50% at 250°C
        int cut = N/2;
        #pragma omp parallel for schedule(static)
        for(int i=0;i<N;i++){
            for(int j=0;j<cut;j++){
                grid[idx(i,j,N)] = 250.0;
            }
        }
    } else {
        // SCENARIO_B: left 25% at 540°C
        int cut = N/4;
        #pragma omp parallel for schedule(static)
        for(int i=0;i<N;i++){
            for(int j=0;j<cut;j++){
                grid[idx(i,j,N)] = 540.0;
            }
        }
    }
}

void write_snapshot_csv(const char* prefix, int iter, const double* grid, int N){
    char fname[256];
    snprintf(fname, sizeof(fname), "%s_iter%06d.csv", prefix, iter);
    FILE* f = fopen(fname, "w");
    if(!f){ perror("fopen snapshot"); return; }
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            fprintf(f, "%s%.6f", (j? ",":""), grid[idx(i,j,N)]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

int main(int argc, char** argv){
    Params p;
    p.N = N_DEFAULT;
    p.max_iter = 10000;
    p.tol = 0.0; // 0 = no convergence check, run until max_iter
    p.scenario = SCENARIO_A;
    p.T_earth = 14.0;
    p.wx = 0.3;
    p.wy = 0.2;
    p.threads = 0; // use OMP_NUM_THREADS
    p.sample_every = 200; // sampling period
    p.dump_snapshots = 1; // save snapshots
    p.metrics_csv = "metrics.csv";
    p.snapshot_prefix = "snapshot";

    // Simple argument parser
    for(int k=1;k<argc;k++){
        if(!strcmp(argv[k], "--b")) p.scenario = SCENARIO_B;
        else if(!strcmp(argv[k], "--a")) p.scenario = SCENARIO_A;
        else if(!strcmp(argv[k], "-N") && k+1<argc) p.N = atoi(argv[++k]);
        else if(!strcmp(argv[k], "-i") && k+1<argc) p.max_iter = atoi(argv[++k]);
        else if(!strcmp(argv[k], "--tol") && k+1<argc) p.tol = atof(argv[++k]);
        else if(!strcmp(argv[k], "--earth") && k+1<argc) p.T_earth = atof(argv[++k]);
        else if(!strcmp(argv[k], "--wx") && k+1<argc) p.wx = atof(argv[++k]);
        else if(!strcmp(argv[k], "--wy") && k+1<argc) p.wy = atof(argv[++k]);
        else if(!strcmp(argv[k], "-t") && k+1<argc) p.threads = atoi(argv[++k]);
        else if(!strcmp(argv[k], "--sample") && k+1<argc) p.sample_every = atoi(argv[++k]);
        else if(!strcmp(argv[k], "--nosnap")) p.dump_snapshots = 0;
        else if(!strcmp(argv[k], "--metrics") && k+1<argc) p.metrics_csv = argv[++k];
        else if(!strcmp(argv[k], "--snap_prefix") && k+1<argc) p.snapshot_prefix = argv[++k];
        else if(!strcmp(argv[k], "-h") || !strcmp(argv[k], "--help")){
            printf(
"Usage: %s [--a|--b] [-N 1024] [-i 10000] [--tol 1e-3] [--earth 14]\n"
"            [--wx 0.3] [--wy 0.2] [-t threads] [--sample 200]\n"
"            [--nosnap] [--metrics metrics.csv] [--snap_prefix snapshot]\n"
"\n"
"(a) average of 4 neighbors; (b) anisotropic: new = wx*(E+W)+wy*(N+S)\n"
"Fixed boundaries; initialization: (a) left 50%% hot=250°C; (b) left 25%% hot=540°C.\n", argv[0]);
            return 0;
        }
    }

    if(p.threads > 0) omp_set_num_threads(p.threads);

    const int N = p.N;
    size_t total = (size_t)N*(size_t)N;
    double* A = (double*) malloc(total*sizeof(double));
    double* B = (double*) malloc(total*sizeof(double));
    if(!A || !B){ fprintf(stderr,"Allocation failed\n"); return 1; }

    init_grid(A, &p);
    // Copy A to B for consistent boundary initialization
    #pragma omp parallel for schedule(static)
    for(size_t q=0;q<total;q++) B[q]=A[q];

    FILE* m = fopen(p.metrics_csv, "w");
    if(!m){ perror("fopen metrics"); return 1; }
    fprintf(m, "iter,max_delta,elapsed_s\n");

    double start = omp_get_wtime();
    double last_log_t = start;
    int iter_done = 0;

    for(int it=1; it<=p.max_iter; ++it){
        double max_delta = 0.0;

        if(p.scenario == SCENARIO_A){
            // new = average of 4 neighbors
            #pragma omp parallel for reduction(max:max_delta) schedule(static) collapse(2)
            for(int i=1;i<N-1;i++){
                for(int j=1;j<N-1;j++){
                    double newv = 0.25 * (
                        A[idx(i-1,j,N)] + A[idx(i+1,j,N)] +
                        A[idx(i,j-1,N)] + A[idx(i,j+1,N)]
                    );
                    double d = fabs(newv - A[idx(i,j,N)]);
                    if(d > max_delta) max_delta = d;
                    B[idx(i,j,N)] = newv;
                }
            }
        } else {
            // Scenario B: anisotropic diffusion
            const double wx = p.wx, wy = p.wy;
            #pragma omp parallel for reduction(max:max_delta) schedule(static) collapse(2)
            for(int i=1;i<N-1;i++){
                for(int j=1;j<N-1;j++){
                    double newv = wx*(A[idx(i,j-1,N)] + A[idx(i,j+1,N)])
                                + wy*(A[idx(i-1,j,N)] + A[idx(i+1,j,N)]);
                    double d = fabs(newv - A[idx(i,j,N)]);
                    if(d > max_delta) max_delta = d;
                    B[idx(i,j,N)] = newv;
                }
            }
        }

        // Copy fixed boundary cells
        #pragma omp parallel for schedule(static)
        for(int j=0;j<N;j++){ B[idx(0,j,N)]   = A[idx(0,j,N)]; B[idx(N-1,j,N)] = A[idx(N-1,j,N)]; }
        #pragma omp parallel for schedule(static)
        for(int i=0;i<N;i++){ B[idx(i,0,N)]   = A[idx(i,0,N)]; B[idx(i,N-1,N)] = A[idx(i,N-1,N)]; }

        // Swap grids
        double* T = A; A = B; B = T;

        double tnow = omp_get_wtime();
        double elapsed = tnow - start;

        // Log metrics every sample_every iterations (and at the end)
        if((p.sample_every > 0 && (it % p.sample_every == 0)) || it==p.max_iter || (p.tol>0 && max_delta < p.tol)){
            fprintf(m, "%d,%.9f,%.6f\n", it, max_delta, elapsed);
            fflush(m);
            if(p.dump_snapshots){
                write_snapshot_csv(p.snapshot_prefix, it, A, N);
            }
            last_log_t = tnow;
        }

        iter_done = it;
        if(p.tol > 0.0 && max_delta < p.tol) break;
    }

    double total_time = omp_get_wtime() - start;
    fclose(m);

    // Print summary
    fprintf(stderr, "Scenario: %s | N=%d | iter=%d | threads=%d | time=%.3fs\n",
        (p.scenario==SCENARIO_A?"A":"B"), N, iter_done,
        (p.threads>0?p.threads:omp_get_max_threads()), total_time);

    free(A); free(B);
    return 0;
}
