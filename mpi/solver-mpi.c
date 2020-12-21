#include <mpi.h>
#include "heat.h"

#define min(a, b) (((a) < (b)) ? (a) : (b))
#define NB 8

/*
 * Blocked Jacobi solver: one iteration step
 */
double relax_jacobi(double *u, double *utmp, unsigned sizex, unsigned sizey)
{
    double diff, sum = 0.0;
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex / nbx;
    nby = NB;
    by = sizey / nby;
    for (int ii = 0; ii < nbx; ii++)
        for (int jj = 0; jj < nby; jj++)
            for (int i = 1 + ii * bx; i <= min((ii + 1) * bx, sizex - 2); i++)
                for (int j = 1 + jj * by; j <= min((jj + 1) * by, sizey - 2); j++)
                {
                    utmp[i * sizey + j] = 0.25 * (u[i * sizey + (j - 1)] + // left
                                                  u[i * sizey + (j + 1)] + // right
                                                  u[(i - 1) * sizey + j] + // top
                                                  u[(i + 1) * sizey + j]); // bottom
                    diff = utmp[i * sizey + j] - u[i * sizey + j];
                    sum += diff * diff;
                }

    return sum;
}

/*
 * Blocked Red-Black solver: one iteration step
 */
double relax_redblack(double *u, unsigned sizex, unsigned sizey)
{
    double unew, diff, sum = 0.0;
    int nbx, bx, nby, by;
    int lsw;

    nbx = NB;
    bx = sizex / nbx;
    nby = NB;
    by = sizey / nby;
    // Computing "Red" blocks
    for (int ii = 0; ii < nbx; ii++)
    {
        lsw = ii % 2;
        for (int jj = lsw; jj < nby; jj = jj + 2)
            for (int i = 1 + ii * bx; i <= min((ii + 1) * bx, sizex - 2); i++)
                for (int j = 1 + jj * by; j <= min((jj + 1) * by, sizey - 2); j++)
                {
                    unew = 0.25 * (u[i * sizey + (j - 1)] + // left
                                   u[i * sizey + (j + 1)] + // right
                                   u[(i - 1) * sizey + j] + // top
                                   u[(i + 1) * sizey + j]); // bottom
                    diff = unew - u[i * sizey + j];
                    sum += diff * diff;
                    u[i * sizey + j] = unew;
                }
    }

    // Computing "Black" blocks
    for (int ii = 0; ii < nbx; ii++)
    {
        lsw = (ii + 1) % 2;
        for (int jj = lsw; jj < nby; jj = jj + 2)
            for (int i = 1 + ii * bx; i <= min((ii + 1) * bx, sizex - 2); i++)
                for (int j = 1 + jj * by; j <= min((jj + 1) * by, sizey - 2); j++)
                {
                    unew = 0.25 * (u[i * sizey + (j - 1)] + // left
                                   u[i * sizey + (j + 1)] + // right
                                   u[(i - 1) * sizey + j] + // top
                                   u[(i + 1) * sizey + j]); // bottom
                    diff = unew - u[i * sizey + j];
                    sum += diff * diff;
                    u[i * sizey + j] = unew;
                }
    }

    return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss(double *u, unsigned sizex, unsigned sizey, int rank, int nproc, int iter, MPI_Comm comm)
{
    MPI_Status status;
    MPI_Request request;
    double unew, diff, sum = 0.0;
    int nbx, bx, nby, by;

    nbx = NB;
    bx = sizex / nbx;
    nby = NB;
    by = sizey / nby;

    // master
    if (rank == 0) {
        for (int ii = 0; ii < nbx; ii++) {
            for (int jj = 0; jj < nby; jj++) {
                for (int i = 1 + ii * bx; i <= min((ii + 1) * bx, sizex - 2); i++)
                    for (int j = 1 + jj * by; j <= min((jj + 1) * by, sizey - 2); j++)
                    {
                        unew = 0.25 * (u[i * sizey + (j - 1)] + // left
                                    u[i * sizey + (j + 1)] + // right
                                    u[(i - 1) * sizey + j] + // top
                                    u[(i + 1) * sizey + j]); // bottom
                        diff = unew - u[i * sizey + j];
                        sum += diff * diff;
                        u[i * sizey + j] = unew;
                    }
                // if it's the last row of the process, send each block downward
                if (nproc > 1 && ii == nbx - 1) {
                    MPI_Isend(&u[(sizex - 2) * sizey + jj * by], by, MPI_DOUBLE, 1, iter, comm, &request);
                }
            }
        }
        // receive new border values from below
        MPI_Recv(&u[(sizex - 1) * sizey], sizey, MPI_DOUBLE, 1, iter, comm, &status);
    }
    // last
    else if (rank == nproc - 1) {
        for (int ii = 0; ii < nbx; ii++) {
            for (int jj = 0; jj < nby; jj++) {
                // wait block from above
                if (ii == 0) {
                    MPI_Recv(&u[jj * by], by, MPI_DOUBLE, rank - 1, iter, comm, &status);
                }
                for (int i = 1 + ii * bx; i <= min((ii + 1) * bx, sizex - 2); i++)
                    for (int j = 1 + jj * by; j <= min((jj + 1) * by, sizey - 2); j++)
                    {
                        unew = 0.25 * (u[i * sizey + (j - 1)] + // left
                                    u[i * sizey + (j + 1)] + // right
                                    u[(i - 1) * sizey + j] + // top
                                    u[(i + 1) * sizey + j]); // bottom
                        diff = unew - u[i * sizey + j];
                        sum += diff * diff;
                        u[i * sizey + j] = unew;
                    }
            }
        }
        // receive new border values from below
        MPI_Send(&u[sizey], sizey, MPI_DOUBLE, rank - 1, iter, comm);
    }
    // rest
    else {
        for (int ii = 0; ii < nbx; ii++) {
            for (int jj = 0; jj < nby; jj++) {
                // wait block from above
                if (ii == 0) {
                    MPI_Recv(&u[jj * by], by, MPI_DOUBLE, rank - 1, iter, comm, &status);
                }
                for (int i = 1 + ii * bx; i <= min((ii + 1) * bx, sizex - 2); i++)
                    for (int j = 1 + jj * by; j <= min((jj + 1) * by, sizey - 2); j++)
                    {
                        unew = 0.25 * (u[i * sizey + (j - 1)] + // left
                                    u[i * sizey + (j + 1)] + // right
                                    u[(i - 1) * sizey + j] + // top
                                    u[(i + 1) * sizey + j]); // bottom
                        diff = unew - u[i * sizey + j];
                        sum += diff * diff;
                        u[i * sizey + j] = unew;
                    }
                // send block bellow
                if (ii == nbx - 1) {
                    MPI_Isend(&u[(sizex - 2) * sizey + jj * by], by, MPI_DOUBLE, rank + 1, iter, comm, &request);
                }
            }
        }
        // send new values above
        MPI_Recv(&u[(sizex - 1) * sizey], sizey, MPI_DOUBLE, rank + 1, iter, comm, &status);
        MPI_Send(&u[sizey], sizey, MPI_DOUBLE, rank - 1, iter, comm);
    }
    return sum;
}
