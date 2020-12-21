#include <stdlib.h>

#include "heat.h"

#define NB 8

#define min(a, b) (((a) < (b)) ? (a) : (b))

// macro with the calculation of a cell in the Gauss function,
// where it will be called multiple times
#define GAUSS_CELL_OPERATION \
unew = 0.25 * (u[i * sizey + (j - 1)] + /* left */\
	u[i * sizey + (j + 1)] + /* right */\
	u[(i - 1) * sizey + j] + /* top */\
	u[(i + 1) * sizey + j]); /* bottom */\
diff = unew - u[i * sizey + j];\
sum += diff * diff;\
u[i * sizey + j]=unew;\

/*
 * Blocked Jacobi solver: one iteration step
 * Instead of calculating the whole matrix at once, it is divided in blocks and
 * calculations are done inside each block.
 */
double relax_jacobi(double *u, double *utmp, unsigned sizex, unsigned sizey)
{
	double diff, sum = 0.0;
	int nbx, bx, nby, by;

	nbx = NB;
	bx = sizex / nbx;
	nby = NB;
	by = sizey / nby;
	/*
 	 * privatize scalar values
 	 *
    */
	// choose block, index X
	// things are shared by default but we must explicitly write them for this exercise
	// use collapse, or the number of threads...
	// loops are balanced (all have more or less the same load) -> static
	/*
	* This function is called many times, the static scheduler preserves locality but the dynamic one doesn't. Create parallel regions from heat-omp.c file.
	* */
// try also to, instead of using reduction, use a performance killer (atomic construct)
#pragma omp parallel for shared(nbx, bx, nby, by, u, utmp, sizex, sizey) private(diff) reduction(+: sum)
	for (int ii = 0; ii < nbx; ii++)
		// choose block, index Y
		for (int jj = 0; jj < nby; jj++)
			// inside the block, choose cell X
			/* not efficient, but possible!
#pragma omp for collapse(2)
 	    */
			for (int i = 1 + ii * bx; i <= min((ii + 1) * bx, sizex - 2); i++)
				// inside the block, choose cell Y
				for (int j = 1 + jj * by; j <= min((jj + 1) * by, sizey - 2); j++) {
					// read u, write utmp/sum -> ok
					utmp[i * sizey + j] = 0.25 * (u[i * sizey + (j - 1)] + // left
												  u[i * sizey + (j + 1)] + // right
												  u[(i - 1) * sizey + j] + // top
												  u[(i + 1) * sizey + j]); // bottom
					diff = utmp[i * sizey + j] - u[i * sizey + j];
					// sum == residual
					// sum is read and written! diff too but it is created in the parallel region so dw
					// sum is a reduction
					//#pragma omp critical
					// critical blocks a region
					//#pragma omp atomic
					// atomic blocks a r/w/u operation
					// atomic is usually more efficient than critical
					sum += diff * diff;
				}
	// barrier is implicit with parallel
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

	// we can use nowait if we maintain the same scheduler
	nbx = NB;
	bx = sizex / nbx;
	nby = NB;
	by = sizey / nby;
	// define ONE parallel region that englobes TWO parallel fors
#pragma omp parallel shared(nbx, bx, nby, by, lsw, u, unew, sizex, sizey) private(diff) reduction(+: sum)
	{
// Computing "Red" blocks
#pragma omp for nowait
		for (int ii = 0; ii < nbx; ii++) {
			lsw = ii % 2;
			for (int jj = lsw; jj < nby; jj = jj + 2)
				for (int i = 1 + ii * bx; i <= min((ii + 1) * bx, sizex - 2); i++)
					for (int j = 1 + jj * by; j <= min((jj + 1) * by, sizey - 2); j++) {
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
#pragma omp for
		for (int ii = 0; ii < nbx; ii++) {
			lsw = (ii + 1) % 2;
			for (int jj = lsw; jj < nby; jj = jj + 2)
				for (int i = 1 + ii * bx; i <= min((ii + 1) * bx, sizex - 2); i++)
					for (int j = 1 + jj * by; j <= min((jj + 1) * by, sizey - 2); j++) {
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
	return sum;
}

/*
 * Blocked Gauss-Seidel solver: one iteration step
 */
double relax_gauss(double *u, unsigned sizex, unsigned sizey)
{
	double unew, diff, sum = 0.0;
	int nbx, bx, nby, by;

	nbx = NB;
	bx = sizex / nbx;
	nby = NB;
	by = sizey / nby;

	int *deps;
	deps = (int *) malloc(nbx * nby * sizeof(int));

	#pragma omp parallel private(diff)
	#pragma omp single
	
	for (int ii = 0; ii < nbx; ii++) 
		for (int jj = 0; jj < nby; jj++) {
			// top left
			if (ii == 0 && jj == 0) {
				#pragma omp task private(unew, diff) depend(out: deps[0])
				for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++) {
					for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
						GAUSS_CELL_OPERATION
					}
				}
			// top
			} else if (ii == 0) {
				//										block to the left		 current block
				#pragma omp task private(unew, diff) depend(in: deps[(jj-1)]) depend(out: deps[jj])
				for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++)
					for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
						GAUSS_CELL_OPERATION
					}
			// left
			} else if (jj == 0) {
				//										block above						current block
				#pragma omp task private(unew, diff) depend(in: deps[(ii-1) * nbx]) depend(out: deps[ii * nbx])
				for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++)
					for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
						GAUSS_CELL_OPERATION
					}
			// others
			} else {
				//												block above					block to the left			current block
				#pragma omp task private(unew, diff) depend(in: deps[(ii-1) * nbx + jj], deps[ii * nbx + (jj-1)]) depend(out: deps[ii * nbx + jj])
				for (int i=1+ii*bx; i<=min((ii+1)*bx, sizex-2); i++)
					for (int j=1+jj*by; j<=min((jj+1)*by, sizey-2); j++) {
						GAUSS_CELL_OPERATION
					}
			}
		}
	// implicit barrier at the end of single
	#pragma omp taskwait
	free(deps);
	return sum;
}

