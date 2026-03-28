#include "hip/hip_runtime.h"
#ifndef __OCFD_SPLIT_H
#define __OCFD_SPLIT_H
#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif
// __global__ void split_Jac3d_Stager_Warming_ker(cudaField d, cudaField u, cudaField v, cudaField w, cudaField cc, cudaSoA fp,
            // cudaSoA fm, cudaField Akx, cudaField Aky, cudaField Akz, cudaJobPackage job);
			
// __global__ void split_Jac3d_Stager_Warming_ker_comp( double pi );
__global__ void split_Jac3d_Stager_Warming_ker_origin(cudaField d, cudaField u, cudaField v, cudaField w, cudaField cc, cudaSoA fp, cudaSoA fm, cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job);

__global__ void split_Jac3d_Stager_Warming_ker(cudaField d, cudaField u, cudaField v, cudaField w, cudaField cc, 
			// cudaField T,
			cudaField Ajac, cudaSoA pdu,
			cudaSoA fpY, cudaSoA fmY,
			cudaSoA fpZ, cudaSoA fmZ,
			cudaField Akx, cudaField Aky, cudaField Akz, 
			cudaField bkx, cudaField bky, cudaField bkz,
			cudaField ckx, cudaField cky, cudaField ckz,
			// dim3 flagxyzb1, dim3 flagxyzb2, dim3 flagxyzb3,
			// dim3 flagxyzb4, dim3 flagxyzb5, dim3 flagxyzb6,
			// cudaJobPackage job
			cudaJobPackage job, double Gamma_d_rcp, double hx_d_rcp, double hy_d_rcp, double hz_d_rcp
			// cudaJobPackage job, cudaField test_x, cudaField test_y, cudaField test_z, cudaSoA test_fp, cudaSoA test_fm
			);
			
__global__ void split_Jac3d_Stager_Warming_ker_boundary(cudaField d, cudaField u, cudaField v, cudaField w, cudaField cc, 
			// cudaField T,
			cudaField Ajac, cudaSoA pdu,
			cudaField Akx, cudaField Aky, cudaField Akz, 
			cudaField bkx, cudaField bky, cudaField bkz,
			cudaField ckx, cudaField cky, cudaField ckz,
			dim3 griddim,
			// cudaJobPackage job
			cudaJobPackage job, double Gamma_d_rcp, double hx_d_rcp, double hy_d_rcp, double hz_d_rcp
			// cudaJobPackage job, cudaField test_x, cudaField test_y, cudaField test_z, cudaField test_fp, cudaField test_fm
			);
			

// __device__ REAL OCFD_weno7_SYMBO_kernel_P_opt1(REAL *stencil); // edit by xx.
// __device__ REAL OCFD_weno7_SYMBO_kernel_M_opt1(REAL *stencil); // edit by xx.

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
