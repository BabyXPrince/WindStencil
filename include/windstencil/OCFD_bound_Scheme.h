#ifndef __OCFD_BOUND_SCHEME_H
#define __OCFD_BOUND_SCHEME_H

#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif


void OCFD_Dx0_bound(cudaField pf , cudaField pfx , cudaJobPackage job_in , dim3 blockdim_in, hipStream_t *stream);
void OCFD_Dy0_bound(cudaField pf , cudaField pfx , cudaJobPackage job_in , dim3 blockdim_in, hipStream_t *stream);
void OCFD_Dz0_bound(cudaField pf , cudaField pfx , cudaJobPackage job_in , dim3 blockdim_in, hipStream_t *stream);


void OCFD_bound(dim3 *flagxyzb, int boundp, int boundm, cudaJobPackage job);

__device__ int OCFD_D0bound_scheme_kernel(REAL* tmp, dim3 flagxyzb, dim3 coords, REAL *stencil, int ka1, cudaJobPackage job);

__device__ int OCFD_bound_scheme_kernel_p(REAL* flag, dim3 flagxyzb, dim3 coords, REAL *stencil, int ka1, int kb1, cudaJobPackage job);
__device__ int OCFD_bound_scheme_kernel_m(REAL* flag, dim3 flagxyzb, dim3 coords, REAL *stencil, int ka1, int kb1, cudaJobPackage job);

	__device__ REAL OCFD_weno5_kernel_P_lift(REAL *stencil); // edit by xx.
	__device__ REAL OCFD_weno5_kernel_P_lift_plus(REAL *stencil); // edit by xx.
	__device__ REAL OCFD_weno5_kernel_P_right_plus(REAL *stencil); // edit by xx.

	__device__ REAL OCFD_weno5_kernel_M_lift(REAL *stencil); // edit by xx.
	__device__ REAL OCFD_weno5_kernel_M_lift_plus(REAL *stencil); // edit by xx.
	__device__ REAL OCFD_weno5_kernel_M_right_plus(REAL *stencil); // edit by xx.
	
		// __device__ REAL OCFD_weno5_kernel_P_lift1(REAL *stencil); // edit by xx.
		// __device__ REAL OCFD_weno5_kernel_P_lift_plus1(REAL *stencil); // edit by xx.
		// __device__ REAL OCFD_weno5_kernel_P_right_plus1(REAL *stencil); // edit by xx.

		// __device__ REAL OCFD_weno5_kernel_M_lift1(REAL *stencil); // edit by xx.
		// __device__ REAL OCFD_weno5_kernel_M_lift_plus1(REAL *stencil); // edit by xx.
		// __device__ REAL OCFD_weno5_kernel_M_right_plus1(REAL *stencil); // edit by xx.

#ifdef __cplusplus
}
#endif
#endif
