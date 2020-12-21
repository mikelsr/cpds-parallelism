/*
 * Iterative solver for heat distribution
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "heat.h"

#include <math.h>
#define MPI_MASTER 0
#define TAG_ITER 40
#define TAG_COLUMNS 41
#define TAG_ALGO 42
#define TAG_U 43
#define TAG_UHELP 44

#define IS_POWER_OF_TWO(x) (x & (x - 1)) == 0
#define PREV_POWER_OF_TWO(x) pow(2, ceil(log(x) / log(2))) / 2

void usage(char* s)
{
    fprintf(stderr,
        "Usage: %s <input file> [result file]\n\n", s);
}

int main(int argc, char* argv[])
{
    unsigned iter;
    FILE *infile, *resfile;
    char* resfilename;
    int nproc, rank;
    MPI_Status status;
    MPI_Request request;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm useful_comm;
    MPI_Group global_group;
    MPI_Comm_group(MPI_COMM_WORLD, &global_group);

    // useful_procs is used when nproc is not a power of 2
    int useful_procs;
    // trim processes to the greatest power of two below nproc
    if (!(IS_POWER_OF_TWO(nproc)) && nproc > 1) {
        MPI_Group useful_group;
        useful_procs = PREV_POWER_OF_TWO(nproc);
        if (rank == MPI_MASTER)
            printf("Number of processes (%d) is not a power of two, it will be reduced to %d processes\n", nproc, useful_procs);

        // create new global group with only useful ranks

        int range[][3] = { useful_procs, nproc - 1, 1 };
        MPI_Group_range_excl(global_group, 1, range, &useful_group);
        MPI_Comm_create(MPI_COMM_WORLD, useful_group, &useful_comm);

        // finish processes that won't be used
        if (useful_comm == MPI_COMM_NULL) {
            printf("Process with rank %d will now finalize\n", rank);
            MPI_Finalize();
            exit(0);
        }
        // update nproc
        nproc = useful_procs;
    }
    else {
        // Otherwise all are useful
        MPI_Comm_create(MPI_COMM_WORLD, global_group, &useful_comm);
    }

    // master process
    if (rank == MPI_MASTER) {
        printf("I am the master (%d) and going to distribute work to %d additional workers ...\n", rank, nproc - 1);

        // algorithmic parameters
        algoparam_t param;
        int np, num_x, num_t;

        int rows, proc_rows, total_proc_rows;

        double runtime, flop;
        double residual = 0.0;
        double total_residual;

        // check arguments
        if (argc < 2) {
            usage(argv[0]);
            return 1;
        }

        // check input file
        if (!(infile = fopen(argv[1], "r"))) {
            fprintf(stderr,
                "\nError: Cannot open \"%s\" for reading.\n\n", argv[1]);

            usage(argv[0]);
            return 1;
        }

        // check result file
        resfilename = (argc >= 3) ? argv[2] : "heat.ppm";

        if (!(resfile = fopen(resfilename, "w"))) {
            fprintf(stderr,
                "\nError: Cannot open \"%s\" for writing.\n\n",
                resfilename);
            usage(argv[0]);
            return 1;
        }

        // check input
        if (!read_input(infile, &param)) {
            fprintf(stderr, "\nError: Error parsing input file.\n\n");
            usage(argv[0]);
            return 1;
        }
        print_params(&param);

        // set the visualization resolution

        param.u = 0;
        param.uhelp = 0;
        param.uvis = 0;
        param.visres = param.resolution;

        if (!initialize(&param)) {
            fprintf(stderr, "Error in Solver initialization.\n\n");
            usage(argv[0]);
            return 1;
        }

        // full size (param.resolution are only the inner points)
        np = param.resolution + 2;

        num_x = param.resolution / nproc;
        num_t = num_x + 2;

        rows = param.resolution;
        proc_rows = rows / nproc;
        total_proc_rows = proc_rows + 2;

        // starting time
        runtime = wtime();

        // send to workers the necessary data to perform computation
        // TODO: Replace invariable messages with broadcast?
        for (int i = 0; i < nproc; i++) {
            if (i > 0) {
                MPI_Send(&param.maxiter, 1, MPI_INT, i, TAG_ITER, useful_comm);
                MPI_Send(&param.resolution, 1, MPI_INT, i, TAG_COLUMNS, useful_comm);
                MPI_Send(&param.algorithm, 1, MPI_INT, i, TAG_ALGO, useful_comm);
                MPI_Send(&param.u[i * num_x * np], (num_t) * (np), MPI_DOUBLE, i, TAG_U, useful_comm);
                MPI_Send(&param.uhelp[i * num_x * np], (num_t) * (np), MPI_DOUBLE, i, TAG_UHELP, useful_comm);
            }
        }

        iter = 0;
        while (1) {
            switch (param.algorithm) {
            case 0: // JACOBI
                residual = relax_jacobi(param.u, param.uhelp, proc_rows + 2, np);
                // Copy uhelp into u
                for (int i = 0; i < np; i++)
                    for (int j = 0; j < np; j++)
                        param.u[i * np + j] = param.uhelp[i * np + j];
                if (nproc > 1) {
                    // send boundary to proc that runs the rows bellow (will have rank 1)
                    MPI_Isend(&param.u[proc_rows * np], np, MPI_DOUBLE, 1, iter, useful_comm, &request);
                    // receive upper boundary of process bellow
                    MPI_Recv(&param.u[(proc_rows + 1) * np], np, MPI_DOUBLE, 1, iter, useful_comm, &status);
                }
                break;
            case 1: // RED-BLACK
                residual = relax_redblack(param.u, np, np);
                break;
            case 2: // GAUSS
                residual = relax_gauss(param.u, total_proc_rows, np, MPI_MASTER, nproc, iter, useful_comm);
                break;
            }

            iter++;

            MPI_Allreduce(&residual, &total_residual, 1, MPI_DOUBLE, MPI_SUM, useful_comm);

            // solution good enough ?
            if (total_residual < 0.00005)
                break;

            // max. iteration reached ? (no limit with maxiter=0)
            if (param.maxiter > 0 && iter >= param.maxiter)
                break;
        }

        // Flop count after iter iterations
        flop = iter * 11.0 * param.resolution * param.resolution;
        // stopping time
        runtime = wtime() - runtime;

        fprintf(stdout, "Time: %04.3f ", runtime);
        fprintf(stdout, "(%3.3f GFlop => %6.2f MFlop/s)\n",
            flop / 1000000000.0,
            flop / runtime / 1000000);
        fprintf(stdout, "Convergence to residual=%f: %d iterations\n", residual, iter);

        // Receive results from other processes
        for (int i = 1; i < nproc; i++) {
            MPI_Recv(&param.u[(proc_rows * i + 1) * np], proc_rows * np, MPI_DOUBLE, i, param.maxiter + 1, useful_comm, &status);
        }

        // for plot...
        coarsen(param.u, np, np,
            param.uvis, param.visres + 2, param.visres + 2);

        write_image(resfile, param.uvis,
            param.visres + 2,
            param.visres + 2);

        finalize(&param);

        fprintf(stdout, "Process %d finished computing with residual value = %f\n", rank, residual);

        MPI_Finalize();

        return 0;
    }
    else {
        printf("I am worker %d and ready to receive work to do ...\n", rank);

        // receive information from master to perform computation locally

        int columns, np, rows;
        int iter, maxiter;
        int algorithm;
        int proc_rows, total_proc_rows;
        double residual, total_residual;

        MPI_Recv(&maxiter, 1, MPI_INT, MPI_MASTER, TAG_ITER, useful_comm, &status);
        MPI_Recv(&columns, 1, MPI_INT, MPI_MASTER, TAG_COLUMNS, useful_comm, &status);
        MPI_Recv(&algorithm, 1, MPI_INT, MPI_MASTER, TAG_ALGO, useful_comm, &status);

        rows = columns;
        np = columns + 2;

        // rows assigned to this process
        proc_rows = rows / nproc;
        total_proc_rows = proc_rows + 2;

        // allocate memory for worker
        // allocate only the rows of this process, the one above and the one bellow
        // first row will be 0 (lwoer border of blocks above)
        // second raw will be np (first real block)
        // almost last raw will be proc_rows * np (last real block)
        // last raw will be (proc_rows + 1) * np (upper border of block below)
        double* u = calloc(sizeof(double), total_proc_rows * np);
        double* uhelp = calloc(sizeof(double), total_proc_rows * np);
        if ((!u) || (!uhelp)) {
            fprintf(stderr, "Error: Cannot allocate memory\n");
            return 0;
        }

        // fill initial values for matrix with values received from master
        MPI_Recv(&u[0], total_proc_rows * (columns + 2), MPI_DOUBLE, MPI_MASTER, TAG_U, useful_comm, &status);
        MPI_Recv(&uhelp[0], total_proc_rows * (columns + 2), MPI_DOUBLE, MPI_MASTER, TAG_UHELP, useful_comm, &status);

        iter = 0;
        while (1) {
            switch (algorithm) {
            case 0: // JACOBI
                residual = relax_jacobi(u, uhelp, total_proc_rows, np);
                // Copy uhelp into u
                // TODO: copy only the parts relevant to this: itself and neighbours
                for (int i = 0; i < total_proc_rows; i++)
                    for (int j = 0; j < np; j++)
                        u[i * np + j] = uhelp[i * np + j];
                // Synchronization: first top-to-bottom, then bottom-to-top
                // Receive border values from the process above
                MPI_Recv(&u[0], np, MPI_DOUBLE, rank - 1, iter, useful_comm, &status);
                // Send border values to the process below (last process does not do this)
                if (rank < nproc - 1) {
                    MPI_Isend(&u[proc_rows * np], np, MPI_DOUBLE, rank + 1, iter, useful_comm, &request);
                    // Receive values from bellow
                    MPI_Recv(&u[(proc_rows + 1) * np], np, MPI_DOUBLE, rank + 1, iter, useful_comm, &status);
                }
                // Send border values to the process above
                MPI_Isend(&u[np], np, MPI_DOUBLE, rank - 1, iter, useful_comm, &request);
                break;
            case 1: // RED-BLACK
                residual = relax_redblack(u, np, np);
                break;
            case 2: // GAUSS
                residual = relax_gauss(u, total_proc_rows, np, rank, nproc, iter, useful_comm);
                if (rank < nproc - 1) {
                    
                }
                //MPI_Send(&u[0], np, MPI_DOUBLE, rank - 1, iter, useful_comm);
                break;
            }
            iter++;

            MPI_Allreduce(&residual, &total_residual, 1, MPI_DOUBLE, MPI_SUM, useful_comm);

            // solution good enough ?
            if (total_residual < 0.00005)
                break;

            // max. iteration reached ? (no limit with maxiter=0)
            if (maxiter > 0 && iter >= maxiter)
                break;
        }

        // Send result to master
        MPI_Send(&u[np], proc_rows * np, MPI_DOUBLE, 0, maxiter + 1, useful_comm);

        if (u)
            free(u);
        if (uhelp)
            free(uhelp);

        fprintf(stdout, "Process %d finished computing %d iterations with residual value = %f\n", rank, iter, residual);

        MPI_Finalize();
        exit(0);
    }
}
