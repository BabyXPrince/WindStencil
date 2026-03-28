#include "hip/hip_runtime.h"
// =============================================================================================
//  含三维Jocabian变换

#include <math.h>
#include "utility.h"
#include "parameters.h"
#include "parameters_d.h"
#include "cuda_commen.h"
#include "cuda_utility.h"
//
#include "OCFD_warp_shuffle.h"
#include "OCFD_Schemes.h"
#include "OCFD_bound_Scheme.h"

#ifdef __cplusplus
extern "C"{
#endif

// #define ny_2lap_d 72
// #define epsl_sw_d 2.2
// #define Gamma_d 1.4
// #define split_C3_d 5.5
// #define split_C1_d 4.4
// #define ny_d 64
// #define nz_d 64

// __device__ REAL OCFD_weno7_SYMBO_kernel_P_opt(REAL *stencil);
// __device__ REAL OCFD_weno7_SYMBO_kernel_M_opt(REAL *stencil);


#define scanY 8
#define scanZ 16

// __global__ void split_Jac3d_Stager_Warming_ker_test(cudaField d, cudaField u, cudaField v, cudaField w,
__global__ void split_Jac3d_Stager_Warming_ker(cudaField d, cudaField u, cudaField v, cudaField w, 
	cudaField cc, 
	// cudaField T,
	cudaField Ajac , cudaSoA du,
	cudaSoA fpY, cudaSoA fmY,
	cudaSoA fpZ, cudaSoA fmZ,
	cudaField Ax, cudaField Ay, cudaField Az,cudaField Bx, cudaField By, cudaField Bz,
	cudaField Cx, cudaField Cy, cudaField Cz, 
	cudaJobPackage job, double Gamma_d_rcp, double hx_d_rcp, double hy_d_rcp, double hz_d_rcp
	// cudaJobPackage job, cudaField test_x, cudaField test_y, cudaField test_z, cudaSoA test_fp, cudaSoA test_fm
	)
{
	// LDS 62.6 KB.
	__shared__ REAL __align__(32)  P_shared[1920]; // y-z-x buffer: 12*4*16 = 768 (6 KB)
	__shared__ REAL __align__(32)  M_shared[1920]; // 768*2.5 = 1920 (15 KB)
	
	__shared__ REAL __align__(128) Trans_shared[1281]; // 256*5=1280 -> 10 KB
	
	__shared__ REAL __align__(64)  FaceY_shared[321]; // 5*4*16=320 -> 2.5 KB
	__shared__ REAL __align__(64)  FaceZ_shared[2561]; // 320*8=2560 -> 20 KB
	
	{
		// eyes on cells WITH LAPs
		// unsigned int x = threadIdx.x + blockIdx.x*blockDim.x + job.start.x;
		// unsigned int y = threadIdx.y + blockIdx.y*blockDim.y + job.start.y;
		// unsigned int z = threadIdx.z + blockIdx.z*blockDim.z + job.start.z;	// without overlalp
		
		unsigned int x = threadIdx.x +       blockIdx.x*(blockDim.x-1) + blockDim.x-1 +job.start.x;	// with x overlap, inner point only.
		unsigned int y = threadIdx.y + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y; // inner point only.
		unsigned int z = threadIdx.z + scanZ*blockIdx.z* blockDim.z                   +job.start.z;
		
		REAL stencil_d[3];  REAL buffer_d[3]; 
		REAL stencil_u[3];  REAL buffer_u[3]; 
		REAL stencil_v[3];  REAL buffer_v[3]; 
		REAL stencil_w[3];  REAL buffer_w[3]; 
		REAL stencil_cc[3]; REAL buffer_cc[3];
		
		REAL stencil_Ax[3]; REAL buffer_Ax[3];
		REAL stencil_Ay[3]; REAL buffer_Ay[3];
		REAL stencil_Az[3]; REAL buffer_Az[3];
		
		REAL Ajacobi;
		REAL rhs[6]; for(int ii=0; ii<6; ii++) rhs[ii] = 0.0; // 0 is dummy, 1~5 is real.
		// REAL rhs1 = 0, rhs2 = 0, rhs3 = 0, rhs4 = 0, rhs5 = 0; // rhs第一步，需要初始化为0，很重要.
		
		// __shared__ REAL  d_shared[257]; // 16*4*4+1
		// __shared__ REAL  u_shared[257];
		// __shared__ REAL  v_shared[257];
		// __shared__ REAL  w_shared[257];
		// __shared__ REAL cc_shared[257];
		
		// __shared__ REAL fp_shared[768];
		// __shared__ REAL fm_shared[768]; // 12 KB
		
		// __shared__ REAL P_shared[1792*2]; // y-z-x buffer: 28*4*16 
		// __shared__ REAL P_shared[1856]; // y-z-x buffer: 28*4*16 = 1792 (14 KB)
		// __shared__ REAL M_shared[1856]; // 23*4*4*5 = 1840 (14.375 KB) -> 1856 (14.5 KB)
		
		// __shared__ REAL Trans_shared[1536]; // 12 KB
		// __shared__ REAL Trans_shared[1281]; // 256*5=1280 -> 10 KB
		
						// ***********************************
							// get_Field_LAP(test_x, x, y, z) = x;
							// get_Field_LAP(test_y, x, y, z) = y;
							// get_Field_LAP(test_z, x, y, z) = z;
						// ***********************************
						
		//======================================
		// 			for X-direction
		//======================================
		// /*
		{	
			{
				// int blk = blockIdx.x / (gridDim.x-1); // the last block.
				// int thd = threadIdx.x / ( job.end.x-job.start.x-(blockDim.x-1)*(gridDim.x-1) ); // the threads which overstep the boundary.
				// int flag = blk*thd; // data prepare without the "if".
				// int flag = x / job.end.x;  // data prepare without the "if".
				// data prepare without the "if".
				int flag1 = (blockDim.x-4 +threadIdx.x) / blockDim.x; // threadIdx.x < 4,  flag = 0; otherwise flag = 1.
				int flag2 = (blockDim.x+12-threadIdx.x) / blockDim.x; // threadIdx.x > 12, flag = 0; otherwise flag = 1.
				
				unsigned int offset = 1 +threadIdx.x+16*threadIdx.y +64*threadIdx.z;
				
				//
				//for rho parameters
				stencil_d[0] = get_Field_LAP(d, x-4, y, z) ;
				stencil_d[1] = get_Field_LAP(d, x+3, y, z) ;
				
					//store inner flow data to lds buffer
					Trans_shared[ (offset -4) *flag1] = stencil_d[0];
					Trans_shared[ (offset +3) *flag2] = stencil_d[1];
				
				//
				//for u parameters
				stencil_u[0] = get_Field_LAP(u, x-4, y, z) ;
				stencil_u[1] = get_Field_LAP(u, x+3, y, z) ;
					
					//store inner flow data to lds buffer 
					Trans_shared[ ( 256 +offset -4) *flag1] = stencil_u[0];
					Trans_shared[ ( 256 +offset +3) *flag2] = stencil_u[1];
				
				//
				//for v parameters
				stencil_v[0] = get_Field_LAP(v, x-4, y, z) ;
				stencil_v[1] = get_Field_LAP(v, x+3, y, z) ;
				
					//store inner flow data to lds buffer 
					Trans_shared[ ( 256*2 +offset -4) *flag1] = stencil_v[0];
					Trans_shared[ ( 256*2 +offset +3) *flag2] = stencil_v[1];
				
				//
				//for w parameters
				stencil_w[0] = get_Field_LAP(w, x-4, y, z) ;
				stencil_w[1] = get_Field_LAP(w, x+3, y, z) ;
					
					//store inner flow data to lds buffer 
					Trans_shared[ ( 256*3 +offset -4) *flag1] = stencil_w[0];
					Trans_shared[ ( 256*3 +offset +3) *flag2] = stencil_w[1];
				
				//
				//for T/cc parameters
				stencil_cc[0] = get_Field_LAP(cc, x-4, y, z);
				stencil_cc[1] = get_Field_LAP(cc, x+3, y, z);
					
					//store inner flow data to lds buffer 
					Trans_shared[ ( 256*4 +offset -4) *flag1] = stencil_cc[0];
					Trans_shared[ ( 256*4 +offset +3) *flag2] = stencil_cc[1];
				
				//
				//for Ax parameters
				stencil_Ax[0] = get_Field_LAP(Ax, x-4, y, z);
				stencil_Ax[1] = get_Field_LAP(Ax, x+3, y, z);
				
				//
				//for Ay parameters
				stencil_Ay[0] = get_Field_LAP(Ay, x-4, y, z);
				stencil_Ay[1] = get_Field_LAP(Ay, x+3, y, z);
				
				//
				//for Az parameters
				stencil_Az[0] = get_Field_LAP(Az, x-4, y, z);
				stencil_Az[1] = get_Field_LAP(Az, x+3, y, z);
				
				//
				//for jacobian parameters
				Ajacobi = get_Field_LAP(Ajac, x, y, z);
				
				// REAL Ajacobi_tmp;
				// Ajacobi_tmp = get_Field_LAP(Ajac, x, y, z); // Ajacobi = get_Field_LAP(Ajac, x-1, y, z);
				// ajac_shared[threadIdx.x+16*threadIdx.y+64*threadIdx.z] = Ajacobi_tmp;
				// Ajacobi = __shfl_up_double(Ajacobi_tmp, 1, hipWarpSize);
			}
			
			// #pragma unroll 1
			// for(int aaa=0; aaa<1000; aaa++)
			{
				// int offset = threadIdx.x+24*threadIdx.y+96*threadIdx.z; // take x as the continuous direction in LDS. x-y-z: 24*4*4 = 384.
				unsigned int offset = threadIdx.x+23*threadIdx.y+92*threadIdx.z; // take x as the continuous direction in LDS. x-y-z: 23*4*4 = 368.
				// REAL ss[3];
				// REAL E1P[3];
				// REAL E2P[3];
				// REAL E3P[3];
				
				#pragma unroll 1
				for(int ii=0; ii<2; ii++)
				{
					//-------------		
					// x-dir
					//-------------
					REAL ss, Ak1, Ak2, Ak3;
					REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
					REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
					
					ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]); // 5
					REAL rcp = 1 / ss;
					Ak1 = stencil_Ax[ii] *rcp;
					Ak2 = stencil_Ay[ii] *rcp;
					Ak3 = stencil_Az[ii] *rcp; // 10*3

					vs = stencil_Ax[ii] * stencil_u[ii] 
					   + stencil_Ay[ii] * stencil_v[ii]
					   + stencil_Az[ii] * stencil_w[ii]; // 5

					E1 = vs;
					E2 = vs - stencil_cc[ii] * ss;
					E3 = vs + stencil_cc[ii] * ss; // 4

					E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
					E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
					E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

					E1M = E1 - E1P;
					E2M = E2 - E2P;
					E3M = E3 - E3P; // 3
					// ----------------------------------------
					tmp0 = stencil_d[ii] *Gamma_d_rcp; // 1
					uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
					uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
					vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
					vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
					wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
					wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3; // 12
					vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
					vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50; // 12
					vv = (Gamma_d - 1.0) * (stencil_u[ii] * stencil_u[ii] 
										  + stencil_v[ii] * stencil_v[ii]
										  + stencil_w[ii] * stencil_w[ii] ); // 7
					W2 = split_C3_d * stencil_cc[ii] * stencil_cc[ii]; // 2
					
					P_shared[     7*ii +offset] = tmp0 * (split_C1_d * E1P + E2P + E3P); // 1
					M_shared[     7*ii +offset] = tmp0 * (split_C1_d * E1M + E2M + E3M);
					P_shared[ 368+7*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_u[ii] + E2P * uc1 + E3P * uc2); // 2
					M_shared[ 368+7*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_u[ii] + E2M * uc1 + E3M * uc2);
					P_shared[ 736+7*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_v[ii] + E2P * vc1 + E3P * vc2); // 3
					M_shared[ 736+7*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_v[ii] + E2M * vc1 + E3M * vc2);
					P_shared[1104+7*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_w[ii] + E2P * wc1 + E3P * wc2); // 4
					M_shared[1104+7*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_w[ii] + E2M * wc1 + E3M * wc2);
					P_shared[1472+7*ii +offset] = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P)); // 5
					M_shared[1472+7*ii +offset] = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
				}
				
				// __threadfence();
				// __syncthreads();
				
				REAL weno_P[8], weno_M[8];
				// #pragma unroll 1
				for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[offset +ii],
										  weno_M[ii] = M_shared[offset +ii];
				
				#pragma unroll 1
				for(int Loop=1; Loop<6; Loop++)
				{
					{
						// x_rhs1+
						REAL tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&weno_P[0]);
						REAL tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
						rhs[Loop] += -Ajacobi*(tmp_r - tmp_l)*hx_d_rcp; // put_du_p_kernel
					}
					for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[368*Loop +offset +ii];
					
					{
						// x_rhs1-
						REAL tmp_l = OCFD_weno7_SYMBO_kernel_M_opt (&weno_M[0]);
						REAL tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
						rhs[Loop] += -Ajacobi*(tmp_r - tmp_l)*hx_d_rcp; // put_du_m_kernel
					}
					for(int ii=0; ii<8; ii++) weno_M[ii] = M_shared[368*Loop +offset +ii];
					
					if(Loop == 2)
					{
						unsigned int offset2 = 1+threadIdx.x+16*threadIdx.y+64*threadIdx.z;
						
						//for rho parameters
						stencil_d[0] = get_Field_LAP(d, x, y-4, z);
						stencil_d[1] = Trans_shared[offset2];
						stencil_d[2] = get_Field_LAP(d, x, y+4, z);

						//for u parameters
						stencil_u[0] = get_Field_LAP(u, x, y-4, z);
						stencil_u[1] = Trans_shared[offset2 +256];
						stencil_u[2] = get_Field_LAP(u, x, y+4, z);
						
						//for v parameters
						stencil_v[0] = get_Field_LAP(v, x, y-4, z);
						stencil_v[1] = Trans_shared[offset2 +256*2];
						stencil_v[2] = get_Field_LAP(v, x, y+4, z);
						
						//for w parameters
						stencil_w[0] = get_Field_LAP(w, x, y-4, z);
						stencil_w[1] = Trans_shared[offset2 +256*3];
						stencil_w[2] = get_Field_LAP(w, x, y+4, z);
						
						//for T/cc parameters
						stencil_cc[0] = get_Field_LAP(cc, x, y-4, z);
						stencil_cc[1] = Trans_shared[offset2 +256*4];
						stencil_cc[2] = get_Field_LAP(cc, x, y+4, z);
						
						//for Ax parameters
						stencil_Ax[0] = get_Field_LAP(Bx, x, y-4, z);
						stencil_Ax[1] = get_Field_LAP(Bx, x, y,   z);
						stencil_Ax[2] = get_Field_LAP(Bx, x, y+4, z);
						
						//for Ay parameters
						stencil_Ay[0] = get_Field_LAP(By, x, y-4, z);
						stencil_Ay[1] = get_Field_LAP(By, x, y,   z);
						stencil_Ay[2] = get_Field_LAP(By, x, y+4, z);
						
						//for Az parameters
						stencil_Az[0] = get_Field_LAP(Bz, x, y-4, z);
						stencil_Az[1] = get_Field_LAP(Bz, x, y,   z);
						stencil_Az[2] = get_Field_LAP(Bz, x, y+4, z);
						// __threadfence();
					}	
					
				}
			}
		}

	// #pragma unroll 1
	// for(int TEST=0; TEST<1000; TEST++)
	// {
		// __syncthreads();
		// /*
		//
		//======================================
		// 			for Y-direction
		//======================================
		{
			{
				y = threadIdx.z + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y;
				z = threadIdx.y + scanZ*blockIdx.z* blockDim.z                   +job.start.z;
				unsigned int offset2 = 1+threadIdx.x+16*threadIdx.z+64*threadIdx.y; // 5
		
				//for rho parameters
				buffer_d[0] = get_Field_LAP(d, x, y, z-4); // 6+1
				buffer_d[1] = Trans_shared[offset2];
				buffer_d[2] = get_Field_LAP(d, x, y, z+4);
				// buffer_d[2] = get_Field_LAP(d, x, y, z+4-flag*8);
				
				//for u parameters
				buffer_u[0] = get_Field_LAP(u, x, y, z-4) ;
				buffer_u[1] = Trans_shared[offset2 +256];
				buffer_u[2] = get_Field_LAP(u, x, y, z+4);
				
				//for v parameters
				buffer_v[0] = get_Field_LAP(v, x, y, z-4) ;
				buffer_v[1] = Trans_shared[offset2 +256*2];
				buffer_v[2] = get_Field_LAP(v, x, y, z+4);
			
				//for w parameters
				buffer_w[0] = get_Field_LAP(w, x, y, z-4) ;
				buffer_w[1] = Trans_shared[offset2 +256*3];
				buffer_w[2] = get_Field_LAP(w, x, y, z+4);
				
				//for T/cc parameters
				buffer_cc[0] = get_Field_LAP(cc, x, y, z-4);
				buffer_cc[1] = Trans_shared[offset2 +256*4];
				buffer_cc[2] = get_Field_LAP(cc, x, y, z+4);
				
				//for Ax parameters
				buffer_Ax[0] = get_Field_LAP(Cx, x, y, z-4) ;
				buffer_Ax[1] = get_Field_LAP(Cx, x, y, z  ) ;
				buffer_Ax[2] = get_Field_LAP(Cx, x, y, z+4);
				
				//for Ay parameters
				buffer_Ay[0] = get_Field_LAP(Cy, x, y, z-4) ;
				buffer_Ay[1] = get_Field_LAP(Cy, x, y, z  ) ;
				buffer_Ay[2] = get_Field_LAP(Cy, x, y, z+4);
				
				//for Az parameters
				buffer_Az[0] = get_Field_LAP(Cz, x, y, z-4) ;
				buffer_Az[1] = get_Field_LAP(Cz, x, y, z  ) ;
				buffer_Az[2] = get_Field_LAP(Cz, x, y, z+4);
			}
			
			// if( threadIdx.x != (blockDim.x-1) )
			// {
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = x;
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = y;
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = z;
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = stencil_w[1];
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = stencil_cc[1];
			// }
			
			// if( x>=job.start.x && x<=job.end.x && y>=job.start.y && y<=job.end.y && z>=job.start.z && z<=job.end.z)
			// if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-5)
			// /*
			// #pragma unroll 1
			// for(int aaa=0; aaa<1000; aaa++)
			{
				y = threadIdx.y + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y;
				z = threadIdx.z + scanZ*blockIdx.z* blockDim.z                   +job.start.z;
				
				unsigned int offset = threadIdx.y+29*threadIdx.x+464*threadIdx.z; // take y as the continuous direction in LDS. y-z-x: 28*4*16.
				REAL ss[3];
				REAL E1P[3];
				REAL E2P[3];
				REAL E3P[3];
				REAL weno_P[8], weno_M[8];
				
				#pragma unroll 1
				for(int ii=0; ii<3; ii++)
				{
					//-------------		
					// y-dir-fpfm-5
					//-------------
					REAL Ak1, Ak2, Ak3;
					REAL vs, E1, E2, E3;
					// REAL E1P, E2P, E3P;
					REAL E1M, E2M, E3M;
					REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
					
					ss[ii] = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]); // 5
					REAL rcp = 1 / ss[ii];
					Ak1 = stencil_Ax[ii] *rcp;
					Ak2 = stencil_Ay[ii] *rcp;
					Ak3 = stencil_Az[ii] *rcp; // 10*3

					vs = stencil_Ax[ii] * stencil_u[ii] 
					   + stencil_Ay[ii] * stencil_v[ii]
					   + stencil_Az[ii] * stencil_w[ii]; // 5

					E1 = vs;
					E2 = vs - stencil_cc[ii] * ss[ii];
					E3 = vs + stencil_cc[ii] * ss[ii]; // 4

					E1P[ii] = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
					E2P[ii] = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
					E3P[ii] = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

					E1M = E1 - E1P[ii];
					E2M = E2 - E2P[ii];
					E3M = E3 - E3P[ii]; // 3
					// ----------------------------------------
					tmp0 = stencil_d[ii] *Gamma_d_rcp; // 1
					uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
					uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
					vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
					vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
					wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
					wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3; // 12
					vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
					vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50; // 12
					vv = (Gamma_d - 1.0) * (stencil_u[ii] * stencil_u[ii] 
										  + stencil_v[ii] * stencil_v[ii]
										  + stencil_w[ii] * stencil_w[ii] ); // 7
					W2 = split_C3_d * stencil_cc[ii] * stencil_cc[ii]; // 2
					
					// take z as the continuous direction in LDS. z-y-x: 28*4*16
					REAL fp5, fm5;
					fp5 = tmp0 * (E1P[ii] * vv + E2P[ii] * vvc1 + E3P[ii] * vvc2 + W2 * (E2P[ii] + E3P[ii]));
					fm5 = tmp0 * (E1M     * vv + E2M     * vvc1 + E3M     * vvc2 + W2 * (E2M     + E3M)); // 18
					
					P_shared[4*ii +offset] = fp5;
					M_shared[4*ii +offset] = fm5;
					
					get_SoA_LAP(fpY, x, y+4*ii-4, z, 4) = fp5; // fpfm_5
					get_SoA_LAP(fmY, x, y+4*ii-4, z, 4) = fm5;
				}
				
				// y-dir-fpfm-1
				// for(int ii=0; ii<3; ii++)
				{
					//-------------		
					// y-dir-fpfm-1
					//-------------
					const int ii = 0;
					// REAL ss;
					REAL vs, E1, E2, E3;
					// REAL E1P, E2P, E3P;
					REAL E1M, E2M, E3M;
					REAL tmp0;
					
					// ss[ii] = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]); // 5

					vs = stencil_Ax[ii] * stencil_u[ii] 
					   + stencil_Ay[ii] * stencil_v[ii]
					   + stencil_Az[ii] * stencil_w[ii]; // 5

					E1 = vs;
					E2 = vs - stencil_cc[ii] * ss[ii];
					E3 = vs + stencil_cc[ii] * ss[ii]; // 4

					// E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
					// E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
					// E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

					E1M = E1 - E1P[ii];
					E2M = E2 - E2P[ii];
					E3M = E3 - E3P[ii]; // 3
					// ----------------------------------------
					tmp0 = stencil_d[ii] *Gamma_d_rcp; // 1
					
					REAL fp1, fm1;
					
					fp1 = tmp0 * (split_C1_d * E1P[ii] + E2P[ii] + E3P[ii]);
					fm1 = tmp0 * (split_C1_d * E1M     + E2M     + E3M); // 8
					
					P_shared[12 +4*ii +offset] = fp1;
					M_shared[12 +4*ii +offset] = fm1;
					
					get_SoA_LAP(fpY, x, y-4, z, 0) = fp1; // fpfm_1
					get_SoA_LAP(fmY, x, y-4, z, 0) = fm1;
				}
				
				__syncthreads();
				
								///////////////////////////
								// buffer_1 prefetching: y_Loop1+: rhs1+
								for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[offset +ii],
														  weno_M[ii] = M_shared[offset +ii];
														  
								// __syncthreads();
								// for(int ii=3; ii<8; ii++) rhs[ii-2] = weno_P[ii];
								// rhs[ii+1] = weno_P[ii], rhs[ii+1] = weno_P[ii], rhs[ii+1] = weno_P[ii], 
								// rhs[ii+1] = weno_P[6], rhs[ii+1] = weno_P[7], ;
				
				// __syncthreads();
				offset -= threadIdx.y;
				REAL Wp_previous = 0.0, Wm_previous = 0.0;
				int point = 1;
				
				#pragma unroll 1
				for(int loop=0; loop<7; loop++)
				{
					// if(loop == 6)
					// {
						// unsigned int offset_tran = threadIdx.z+4*threadIdx.y+16*threadIdx.x;
						// Trans_shared[      offset_tran] = rhs[1];
						// Trans_shared[256  +offset_tran] = rhs[2];
						// Trans_shared[512  +offset_tran] = rhs[3];
						// Trans_shared[768  +offset_tran] = rhs[4];
						// Trans_shared[1024 +offset_tran] = rhs[5];
					// }
					
					#pragma unroll 1
					for(int nn=0; nn<2; nn++)
					{
						// if(loop == 6) break;
						if(loop+nn >= 6) break;
						
						//-------------
						// y-dir-1234
						//-------------
						
						int u3Bits = ( 0b110110110010001000100011010001110000101100 >> 3*(loop*2+nn) ) & 0b111;
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("LDS write：%d:  %d\n", (loop*2+nn), u3Bits);
						
						// int u2Bits_2 = ( 0b1001001001001001001001 >> 2*(loop*2+nn) ) & 0b11;
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("position of grid：%d:  %d\n", (loop*2+nn), u2Bits_2);
						
						// int u2Bits_1 = ( 0b1111111010100101010000 >> 2*(loop*2+nn) ) & 0b11;
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("formula of fpfm：%d:  %d\n", (loop*2+nn), u2Bits_1);
						
						// int ii = u2Bits_2;
						int ii = point % 3; // 1,2; 0,1; 2,0; 1,2; 0,1; 2,x; x,x;
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("position of grid：%d:  %d\n", (loop*2+nn), ii);
						
						REAL Ak1, Ak2, Ak3;
						REAL vs, E1, E2, E3;
						// REAL E1P, E2P, E3P;
						REAL E1M, E2M, E3M;
						REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2;
						
						// ss[ii] = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]); // 5
						REAL rcp = 1 / ss[ii];
						Ak1 = stencil_Ax[ii] *rcp;
						Ak2 = stencil_Ay[ii] *rcp;
						Ak3 = stencil_Az[ii] *rcp; // 10*3

						vs = stencil_Ax[ii] * stencil_u[ii] 
						   + stencil_Ay[ii] * stencil_v[ii]
						   + stencil_Az[ii] * stencil_w[ii]; // 5

						E1 = vs;
						E2 = vs - stencil_cc[ii] * ss[ii];
						E3 = vs + stencil_cc[ii] * ss[ii]; // 4

						// E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 5*3

						E1M = E1 - E1P[ii];
						E2M = E2 - E2P[ii];
						E3M = E3 - E3P[ii]; // 3
						// ----------------------------------------
						tmp0 = stencil_d[ii] *Gamma_d_rcp;
						uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
						uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
						vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
						vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
						wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
						wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3; // 6*2+1
						
						double N1[4] = {1.0, stencil_u[ii], stencil_v[ii], stencil_w[ii]};
						double N2[4] = {1.0, uc1, vc1, wc1};
						double N3[4] = {1.0, uc2, vc2, wc2};
						
						// int fml = u2Bits_1; // formula
						int fml = point / 3; // formula: 0,0; 
						point += 1;
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("formula of fpfm：%d:  %d\n", (loop*2+nn), fml);

						REAL fp, fm;
						fp = tmp0 * ( N1[fml]*split_C1_d*E1P[ii]  +  N2[fml]*E2P[ii]  +  N3[fml]*E3P[ii] );
						fm = tmp0 * ( N1[fml]*split_C1_d*E1M      +  N2[fml]*E2M      +  N3[fml]*E3M );// 7*2
						
						P_shared[4*u3Bits +threadIdx.y+offset] = fp;
						M_shared[4*u3Bits +threadIdx.y+offset] = fm;
						
						get_SoA_LAP(fpY, x, y+4*ii-4, z, fml) = fp;
						get_SoA_LAP(fmY, x, y+4*ii-4, z, fml) = fm;
						
									// if(blockIdx.x != (gridDim.x-1) && blockIdx.y != (gridDim.y-1) && blockIdx.z != (gridDim.z-1) 
									// && threadIdx.x != (blockDim.x-1) && threadIdx.y != (blockDim.y-1) && threadIdx.z != (blockDim.z-1))
										// printf("lds offset: %d, threadIdx.y=%d\n", 4*u3Bits +threadIdx.y+offset, threadIdx.y );
					}
					
					// if(loop == 5)
					
					// __syncthreads();
					// __threadfence();
					
					uint32_t rhs_flag[4] = {0b1001000001001001000001000100, 0b1010000001001010100100000100,
											0b1010000001001010101000000100, 0b1010100100001010101010010000};
					int rhs_flag_w = ( rhs_flag[threadIdx.y] >> (loop*4+3) ) & 0b1;
					int rhs_flag_D = ( rhs_flag[threadIdx.y] >> (loop*4+2) ) & 0b1;
					int rhs_flag_U = ( rhs_flag[threadIdx.y] >> (loop*4+1) ) & 0b1;
					int rhs_flag_P = ( rhs_flag[threadIdx.y] >> (loop*4  ) ) & 0b1;
					// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.z == 0)
						// printf("loop: %d, rhs_flag of threadIdx.y=%d： %d,%d,%d,%d\n", loop+1, threadIdx.y, rhs_flag_w, rhs_flag_D, rhs_flag_U, rhs_flag_P );
					
					uint32_t rhs_ID[4] = {0b100000011010000001101, 0b100000011010001000101,
										  0b100000011010001000101, 0b100011000010001101000};
					int rhs_id = ( rhs_ID[threadIdx.y] >> (loop*3) ) & 0b111;
					// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.z == 0)
						// printf("loop: %d, rhs_id of threadIdx.y=%d: %d\n", loop+1, threadIdx.y, rhs_id);
							
							// if(loop == 1)
							// {
								// for(int ii=0; ii<5; ii++) rhs[ii+1] = weno_P[ii];
								// for(int ii=3; ii<8; ii++) rhs[ii-2] = weno_P[ii];
							// }
							// __syncthreads();
							// if(loop == 6 && threadIdx.y == 0)
							// if(loop == 6)
							// {
								// for(int ii=0; ii<5; ii++) rhs[ii+1] = weno_M[ii];
								// for(int ii=3; ii<8; ii++) rhs[ii-2] = weno_P[ii];
							// }
					
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.z == 0)
								// printf("loop: %d, threadIdx.y=%d, rhs5 before = ：%lf\n", loop+1, threadIdx.y, rhs[5] );
							
					// if(loop == 5)
					
					
					{
						// y_rhs+
						REAL weno = OCFD_weno7_SYMBO_kernel_P_opt(&weno_P[0]);
						REAL DOWN = __shfl_down_double(weno, 16, hipWarpSize);
						REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
						rhs[rhs_id] += -Ajacobi*( (2*rhs_flag_w-1)*weno +rhs_flag_D*DOWN -rhs_flag_U*UP -rhs_flag_P*Wp_previous )*hy_d_rcp; // 10
						
								// if(loop == 6 && threadIdx.y != 0) rhs[1] = ;
								// if(loop == 6 ) rhs[1] = weno;
								// if(loop == 6 ) rhs[2] = Wp_previous;
								// if(loop == 6 ) rhs[3] = Ajacobi;
								// if(loop == 6 ) rhs[4] = -Ajacobi*( (2*rhs_flag_w-1)*weno +rhs_flag_D*DOWN -rhs_flag_U*UP -rhs_flag_P*Wp_previous )*hy_d_rcp;
								
						Wp_previous = weno;
					}
					
					{
						// y_rhs-
						REAL weno = OCFD_weno7_SYMBO_kernel_M_opt(&weno_M[0]);
						REAL DOWN = __shfl_down_double(weno, 16, hipWarpSize);
						REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
						rhs[rhs_id] += -Ajacobi*( (2*rhs_flag_w-1)*weno +rhs_flag_D*DOWN -rhs_flag_U*UP -rhs_flag_P*Wm_previous )*hy_d_rcp;
						
								// if(loop == 6) rhs[1] = ;
								// if(loop == 6) rhs[1] = weno;
								// if(loop == 6) rhs[2] = Wm_previous;
								// if(loop == 6) rhs[3] = Ajacobi;
								// if(loop == 6) rhs[4] = -Ajacobi*( (2*rhs_flag_w-1)*weno +rhs_flag_D*DOWN -rhs_flag_U*UP -rhs_flag_P*Wm_previous )*hy_d_rcp;
								
						Wm_previous = weno;
					}
					
					int u3Face = ( 0b011010000001000100000 >> 3*loop ) & 0b111; // 0,5,1,2,0,3,4
					int flag_face = ( 0b1101110 >> loop ) & 0b1; // 0,5,1,2,0,3,4
					int flag_thread = threadIdx.y / 3; // 0,0,0,1
						// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("Face flag：%d:  %d, %d\n", loop, u3Face, flag_face);
					int offset_face = 1+u3Face*64 +threadIdx.x+16*threadIdx.z; // take x as the continuous direction in LDS. x-z: 16*4.
					FaceY_shared[offset_face *flag_face *flag_thread] = Wp_previous + Wm_previous;
						// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0) printf("Face flag：%d:  %d\n", loop, offset_face *flag_face *flag_thread);
					
					// unsigned int offset_tran = threadIdx.z+4*threadIdx.y+16*threadIdx.x;
					// Trans_shared[256*rhs_id +offset_tran] = rhs[rhs_id];
					
					// __syncthreads();
					// uint8_t offset_lds[8];
					uint64_t lds_load[7][4] = {
						// {0b0011100110001010010000011000100000100000, 0b0100000111001100010100100000110001000001, 0b0100101000001110011000101001000001100010, 0b0101001001010000011100110001010010000011}, // 1
						{0b1001110010100011000001111011100110101100, 0b1010010011100101000110000011110111001101, 
						 0, 0b0101101010010010100000111001100010100100}, // 1
						{0b1101111010110011100000011000100000100000, 0b1010110100100111001010001100000111101110,
						 0b1011010101101001001110010100011000001111, 0b1011110110101011010010011100101000110000}, // 2
						{0b0010011011110101100111000000110001000001, 0b0010100100110111101011001110000001100010,
						 0b0011000101001001101111010110011100000011, 0b0011100110001010010011011110101100111000}, // 3
						{0b0111101110011010110001011010100100101000, 0b1000001111011100110101100010110101001001,
						 0b1000110000011110111001101011000101101010, 0b1001010001100000111101110011010110001011}, // 4
						{0b0011100110001010010000011000100000100000, 0,
						 0 , 0b1001110010100011000001111011100110101100}, // 5
						{0b0100000111001100010100100000110001000001, 0b0100101000001110011000101001000001100010,
						 0b0101001001010000011100110001010010000011, 0b0101101010010010100000111001100010100100}, // 6
						{0, 0, 0, 0} // 7
					};
					
					// #pragma unroll 2
					for(int ii=0; ii<8; ii++)
					{
						if(loop == 6) break;
						// offset_lds[ii] = ( lds_load[loop][threadIdx.y] >> (ii * 5) ) & 0b11111;
						int offset_LDS = ( lds_load[loop][threadIdx.y] >> (ii * 5) ) & 0b11111;
						weno_P[ii] = P_shared[offset_LDS +offset];
						weno_M[ii] = M_shared[offset_LDS +offset];
						
									// if(blockIdx.x == (gridDim.x-1) && blockIdx.y == (gridDim.y-1) && blockIdx.z == (gridDim.z-1) 
									// && threadIdx.x == (blockDim.x-1) && threadIdx.y == (blockDim.y-1) && threadIdx.z == (blockDim.z-1))
											// printf("lds offset: %d, threadIdx.y=%d\n", 1792 +offset_LDS +offset, threadIdx.y );
					}
					
					// if(loop == 5)
					
					// __syncthreads();
					// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.z == 0)
						// printf("loop: %d, loding offset of LDS of threadIdx.y=%d： %d,%d,%d,%d,%d,%d,%d,%d\n", loop+1, threadIdx.y, offset_lds[0], offset_lds[1], offset_lds[2], offset_lds[3], offset_lds[4], offset_lds[5], offset_lds[6], offset_lds[7] );
					
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.z == 0)
								// printf("loop: %d, threadIdx.y=%d, rhs5 after = ：%lf\n", loop+1, threadIdx.y, rhs[5] );
					
					
				} // loop-7
				
				// __syncthreads();
				
			}
			// */
		}
		// */
		
		
		//======================================
		// 			for Z-direction
		//======================================
		// /*
		{
			y = threadIdx.z + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y;
			z = threadIdx.y + scanZ*blockIdx.z* blockDim.z                   +job.start.z;
			
			REAL RHS[6]; for(int ii=0; ii<6; ii++) RHS[ii] = 0.0;
			{
				unsigned int offset_tran = threadIdx.z+4*threadIdx.y+16*threadIdx.x;
				Trans_shared[      offset_tran] = rhs[1];
				Trans_shared[256  +offset_tran] = rhs[2];
				Trans_shared[512  +offset_tran] = rhs[3];
				Trans_shared[768  +offset_tran] = rhs[4];
				Trans_shared[1024 +offset_tran] = rhs[5];
				
				Ajacobi = get_Field_LAP(Ajac, x, y, z);
				// __syncthreads();
				// __threadfence();
					// __asm__(
					// " s_waitcnt lgkmcnt(0)  \n\t"
					// " s_waitcnt vmcnt(0)  \n\t"
					// );
			}
			
			// #pragma unroll 1
			// for(int aaa=0; aaa<1000; aaa++)
			{
				unsigned int offset = threadIdx.y+29*threadIdx.x+464*threadIdx.z; // take y as the continuous direction in LDS. y-z-x: 28*4*16.
				REAL ss[3];
				REAL E1P[3];
				REAL E2P[3];
				REAL E3P[3];
				REAL weno_P[8], weno_M[8];
				
				#pragma unroll 1
				for(int ii=0; ii<3; ii++)
				{
					//-------------		
					// z-dir-fpfm-5
					//-------------
					REAL Ak1, Ak2, Ak3;
					REAL vs, E1, E2, E3;
					// REAL E1P, E2P, E3P;
					REAL E1M, E2M, E3M;
					REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
					
					ss[ii] = sqrt(buffer_Ax[ii]*buffer_Ax[ii] + buffer_Ay[ii]*buffer_Ay[ii] + buffer_Az[ii]*buffer_Az[ii]); // 5
					
					REAL rcp = 1 / ss[ii];
					Ak1 = buffer_Ax[ii] *rcp;
					Ak2 = buffer_Ay[ii] *rcp;
					Ak3 = buffer_Az[ii] *rcp; // 10*3

					vs = buffer_Ax[ii] * buffer_u[ii] 
					   + buffer_Ay[ii] * buffer_v[ii]
					   + buffer_Az[ii] * buffer_w[ii]; // 5

					E1 = vs;
					E2 = vs - buffer_cc[ii] * ss[ii];
					E3 = vs + buffer_cc[ii] * ss[ii]; // 4

					E1P[ii] = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
					E2P[ii] = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
					E3P[ii] = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

					E1M = E1 - E1P[ii];
					E2M = E2 - E2P[ii];
					E3M = E3 - E3P[ii]; // 3
					// ----------------------------------------
					tmp0 = buffer_d[ii] *Gamma_d_rcp; // 1
					uc1  = buffer_u[ii] - buffer_cc[ii] * Ak1;
					uc2  = buffer_u[ii] + buffer_cc[ii] * Ak1;
					vc1  = buffer_v[ii] - buffer_cc[ii] * Ak2;
					vc2  = buffer_v[ii] + buffer_cc[ii] * Ak2;
					wc1  = buffer_w[ii] - buffer_cc[ii] * Ak3;
					wc2  = buffer_w[ii] + buffer_cc[ii] * Ak3; // 12
					vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
					vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50; // 12
					vv = (Gamma_d - 1.0) * (buffer_u[ii] * buffer_u[ii] 
										  + buffer_v[ii] * buffer_v[ii]
										  + buffer_w[ii] * buffer_w[ii] ); // 7
					W2 = split_C3_d * buffer_cc[ii] * buffer_cc[ii]; // 2
					
					// take z as the continuous direction in LDS. z-y-x: 28*4*16
					REAL fp5, fm5;
					fp5 = tmp0 * (E1P[ii] * vv + E2P[ii] * vvc1 + E3P[ii] * vvc2 + W2 * (E2P[ii] + E3P[ii]));
					fm5 = tmp0 * (E1M     * vv + E2M     * vvc1 + E3M     * vvc2 + W2 * (E2M     + E3M)); // 18
					
					P_shared[4*ii +offset] = fp5;
					M_shared[4*ii +offset] = fm5;
					
					get_SoA_LAP(fpZ, x, y, z+4*ii-4, 4) = fp5; // fpfm_5
					get_SoA_LAP(fmZ, x, y, z+4*ii-4, 4) = fm5;
				}
				
				// z-dir-fpfm-1
				// for(int ii=0; ii<3; ii++)
				{
					//-------------		
					// z-dir-fpfm-1
					//-------------
					const int ii = 0;
					// REAL ss;
					REAL vs, E1, E2, E3;
					// REAL E1P, E2P, E3P;
					REAL E1M, E2M, E3M;
					REAL tmp0;
					
					// ss[ii] = sqrt(buffer_Ax[ii]*buffer_Ax[ii] + buffer_Ay[ii]*buffer_Ay[ii] + buffer_Az[ii]*buffer_Az[ii]); // 5

					vs = buffer_Ax[ii] * buffer_u[ii] 
					   + buffer_Ay[ii] * buffer_v[ii]
					   + buffer_Az[ii] * buffer_w[ii]; // 5

					E1 = vs;
					E2 = vs - buffer_cc[ii] * ss[ii];
					E3 = vs + buffer_cc[ii] * ss[ii]; // 4

					// E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
					// E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
					// E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

					E1M = E1 - E1P[ii];
					E2M = E2 - E2P[ii];
					E3M = E3 - E3P[ii]; // 3
					// ----------------------------------------
					tmp0 = buffer_d[ii] *Gamma_d_rcp; // 1
					
					REAL fp1, fm1;
					fp1 = tmp0 * (split_C1_d * E1P[ii] + E2P[ii] + E3P[ii]);
					fm1 = tmp0 * (split_C1_d * E1M     + E2M     + E3M); // 8
					
					P_shared[12 +4*ii +offset] = fp1;
					M_shared[12 +4*ii +offset] = fm1;
					
					get_SoA_LAP(fpZ, x, y, z-4, 0) = fp1; // fpfm_1
					get_SoA_LAP(fmZ, x, y, z-4, 0) = fm1;
				}
				
				__syncthreads();
								///////////////////////////
								// buffer_1 prefetching: y_Loop1+: rhs1+
								// #pragma unroll 1
								for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[offset +ii],
														  weno_M[ii] = M_shared[offset +ii];
								
								// for(int ii=4; ii<8; ii++) weno_P[ii] = P_shared[offset +ii],
														  // weno_M[ii] = M_shared[offset +ii];
														  
								// unsigned int offset_tran = threadIdx.y+4*threadIdx.z+16*threadIdx.x;
								// rhs[1] = Trans_shared[      offset_tran] ;
								// rhs[2] = Trans_shared[256  +offset_tran] ;
								// rhs[3] = Trans_shared[512  +offset_tran] ;
								// rhs[4] = Trans_shared[768  +offset_tran] ;
								// rhs[5] = Trans_shared[1024 +offset_tran] ;
								
				
				// __syncthreads();
				// __threadfence();
				offset -= threadIdx.y;
				REAL Wp_previous = 0.0, Wm_previous = 0.0;
				int point = 1;
				
				#pragma unroll 1
				for(int loop=0; loop<7; loop++)
				{
					// if(loop == 0)
					// {
						// for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[offset +threadIdx.y +ii],
												  // weno_M[ii] = M_shared[offset +threadIdx.y +ii];
						
						// unsigned int offset_tran = threadIdx.y+4*threadIdx.z+16*threadIdx.x;
						// rhs[1] = Trans_shared[256  +offset_tran] ;
						// rhs[2] = Trans_shared[512  +offset_tran] ;
						// rhs[3] = Trans_shared[768  +offset_tran] ;
						// rhs[4] = Trans_shared[1024 +offset_tran] ;
						// rhs[5] = Trans_shared[1280 +offset_tran] ;
					// }
					
					#pragma unroll 1
					for(int nn=0; nn<2; nn++)
					{
						// if(loop == 6) break;
						if(loop+nn >= 6) break;
						
						//-------------
						// z-dir-1234
						//-------------
						
						int u3Bits = ( 0b110110110010001000100011010001110000101100 >> 3*(loop*2+nn) ) & 0b111;
						
						// int u2Bits_2 = ( 0b1001001001001001001001 >> 2*(loop*2+nn) ) & 0b11;
						
						// int u2Bits_1 = ( 0b1111111010100101010000 >> 2*(loop*2+nn) ) & 0b11;
						
						// int ii = u2Bits_2;
						int ii = point % 3;
						
						REAL Ak1, Ak2, Ak3;
						REAL vs, E1, E2, E3;
						// REAL E1P, E2P, E3P;
						REAL E1M, E2M, E3M;
						REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2;
						
						// ss[ii] = sqrt(buffer_Ax[ii]*buffer_Ax[ii] + buffer_Ay[ii]*buffer_Ay[ii] + buffer_Az[ii]*buffer_Az[ii]); // 5
						REAL rcp = 1 / ss[ii];
						Ak1 = buffer_Ax[ii] *rcp;
						Ak2 = buffer_Ay[ii] *rcp;
						Ak3 = buffer_Az[ii] *rcp; // 10*3

						vs = buffer_Ax[ii] * buffer_u[ii] 
						   + buffer_Ay[ii] * buffer_v[ii]
						   + buffer_Az[ii] * buffer_w[ii]; // 5

						E1 = vs;
						E2 = vs - buffer_cc[ii] * ss[ii];
						E3 = vs + buffer_cc[ii] * ss[ii]; // 4

						// E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 5*3

						E1M = E1 - E1P[ii];
						E2M = E2 - E2P[ii];
						E3M = E3 - E3P[ii]; // 3
						// ----------------------------------------
						tmp0 = buffer_d[ii] *Gamma_d_rcp;
						uc1  = buffer_u[ii] - buffer_cc[ii] * Ak1;
						uc2  = buffer_u[ii] + buffer_cc[ii] * Ak1;
						vc1  = buffer_v[ii] - buffer_cc[ii] * Ak2;
						vc2  = buffer_v[ii] + buffer_cc[ii] * Ak2;
						wc1  = buffer_w[ii] - buffer_cc[ii] * Ak3;
						wc2  = buffer_w[ii] + buffer_cc[ii] * Ak3; // 6*2+1
						
						double N1[4] = {1.0, buffer_u[ii], buffer_v[ii], buffer_w[ii]};
						double N2[4] = {1.0, uc1, vc1, wc1};
						double N3[4] = {1.0, uc2, vc2, wc2};
						
						// int fml = u2Bits_1; // formula
						int fml = point / 3; // formula
						point += 1;
						
						REAL fp, fm;
						fp = tmp0 * ( N1[fml]*split_C1_d*E1P[ii]  +  N2[fml]*E2P[ii]  +  N3[fml]*E3P[ii] );
						fm = tmp0 * ( N1[fml]*split_C1_d*E1M      +  N2[fml]*E2M      +  N3[fml]*E3M );// 7*2
						
						P_shared[4*u3Bits +threadIdx.y+offset] = fp;
						M_shared[4*u3Bits +threadIdx.y+offset] = fm;
						
						get_SoA_LAP(fpZ, x, y, z+4*ii-4, fml) = fp;
						get_SoA_LAP(fmZ, x, y, z+4*ii-4, fml) = fm;
					}
					
					// __syncthreads();
					uint32_t rhs_flag[4] = {0b1001000001001001000001000100, 0b1010000001001010100100000100,
											0b1010000001001010101000000100, 0b1010100100001010101010010000};
					int rhs_flag_w = ( rhs_flag[threadIdx.y] >> (loop*4+3) ) & 0b1;
					int rhs_flag_D = ( rhs_flag[threadIdx.y] >> (loop*4+2) ) & 0b1;
					int rhs_flag_U = ( rhs_flag[threadIdx.y] >> (loop*4+1) ) & 0b1;
					int rhs_flag_P = ( rhs_flag[threadIdx.y] >> (loop*4  ) ) & 0b1;
					
					uint32_t rhs_ID[4] = {0b100000011010000001101, 0b100000011010001000101,
										  0b100000011010001000101, 0b100011000010001101000};
					int rhs_id = ( rhs_ID[threadIdx.y] >> (loop*3) ) & 0b111;
					{
						// z_rhs+
						REAL weno = OCFD_weno7_SYMBO_kernel_P_opt(&weno_P[0]);
						REAL DOWN = __shfl_down_double(weno, 16, hipWarpSize);
						REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
						RHS[rhs_id] += -Ajacobi*( (2*rhs_flag_w-1)*weno +rhs_flag_D*DOWN -rhs_flag_U*UP -rhs_flag_P*Wp_previous )*hz_d_rcp; // 10
						Wp_previous = weno;
					}
					
					{
						// z_rhs-
						REAL weno = OCFD_weno7_SYMBO_kernel_M_opt(&weno_M[0]);
						REAL DOWN = __shfl_down_double(weno, 16, hipWarpSize);
						REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
						RHS[rhs_id] += -Ajacobi*( (2*rhs_flag_w-1)*weno +rhs_flag_D*DOWN -rhs_flag_U*UP -rhs_flag_P*Wm_previous )*hz_d_rcp;
						Wm_previous = weno;
					}
					
					int u3Face = ( 0b011010000001000100000 >> 3*loop ) & 0b111; // 0,5,1,2,0,3,4  #(-1)
					int flag_face = ( 0b1101110 >> loop ) & 0b1; // 0,1,1,1,0,1,1
					int flag_thread = threadIdx.y / 3; // 0,0,0,1
					int offset_face = 1+u3Face*64 +threadIdx.x+16*threadIdx.z; // take x as the continuous direction in LDS. x-z: 16*4.
					FaceZ_shared[offset_face *flag_face *flag_thread] = Wp_previous + Wm_previous;
					
					// __syncthreads();
					uint64_t lds_load[7][4] = {
						// {0b0011100110001010010000011000100000100000, 0b0100000111001100010100100000110001000001, 0b0100101000001110011000101001000001100010, 0b0101001001010000011100110001010010000011}, // 1
						{0b1001110010100011000001111011100110101100, 0b1010010011100101000110000011110111001101, 
						 0, 0b0101101010010010100000111001100010100100}, // 1
						{0b1101111010110011100000011000100000100000, 0b1010110100100111001010001100000111101110,
						 0b1011010101101001001110010100011000001111, 0b1011110110101011010010011100101000110000}, // 2
						{0b0010011011110101100111000000110001000001, 0b0010100100110111101011001110000001100010,
						 0b0011000101001001101111010110011100000011, 0b0011100110001010010011011110101100111000}, // 3
						{0b0111101110011010110001011010100100101000, 0b1000001111011100110101100010110101001001,
						 0b1000110000011110111001101011000101101010, 0b1001010001100000111101110011010110001011}, // 4
						{0b0011100110001010010000011000100000100000, 0,
						 0 , 0b1001110010100011000001111011100110101100}, // 5
						{0b0100000111001100010100100000110001000001, 0b0100101000001110011000101001000001100010,
						 0b0101001001010000011100110001010010000011, 0b0101101010010010100000111001100010100100}, // 6
						{0, 0, 0, 0} // 7
					};
					
					// #pragma unroll 1
					for(int ii=0; ii<8; ii++)
					{
						if(loop == 6) break;
						int offset_LDS = ( lds_load[loop][threadIdx.y] >> (ii * 5) ) & 0b11111;
						weno_P[ii] = P_shared[offset_LDS +offset];
						weno_M[ii] = M_shared[offset_LDS +offset];
					}
					
					// __syncthreads();
					
					// if( threadIdx.x != (blockDim.x-1) && loop > 1)
					// {
						// int GLB[5] = {4,0,1,2,3};
						// int num = GLB[loop-2]; // loop-2: 0,1,2,3,4
						// unsigned int offset_tran = threadIdx.y+4*threadIdx.z+16*threadIdx.x;
						// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, num) = RHS[num+1] +Trans_shared[256*num+offset_tran];
					// }
					
					// if(loop == 5)
					// {
						// unsigned int offset_tran = threadIdx.y+4*threadIdx.z+16*threadIdx.x;
						// rhs[1] = Trans_shared[      offset_tran] ;
						// rhs[2] = Trans_shared[256  +offset_tran] ;
						// rhs[3] = Trans_shared[512  +offset_tran] ;
						// rhs[4] = Trans_shared[768  +offset_tran] ;
						// rhs[5] = Trans_shared[1024 +offset_tran] ;
					// }
				}
				
			}
			
			unsigned int offset_tran = threadIdx.y+4*threadIdx.z+16*threadIdx.x;
			if( threadIdx.x != (blockDim.x-1) )
			{
				get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = RHS[1] +Trans_shared[      offset_tran];
				get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = RHS[2] +Trans_shared[256  +offset_tran];
				get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = RHS[3] +Trans_shared[512  +offset_tran];
				get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = RHS[4] +Trans_shared[768  +offset_tran];
				get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = RHS[5] +Trans_shared[1024 +offset_tran];
				
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = RHS[1] +rhs[1];
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = RHS[2] +rhs[2];
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = RHS[3] +rhs[3];
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = RHS[4] +rhs[4];
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = RHS[5] +rhs[5];
			}
		}
		// */
		// __syncthreads();
	// }

		// if( threadIdx.x != (blockDim.x-1) && threadIdx.y <2 )
		// if( threadIdx.x != (blockDim.x-1) )
		// {
			// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = rhs[1];
			// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = rhs[2];
			// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = rhs[3];
			// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = rhs[4];
			// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = rhs[5];
		// }
		
	}
	
	
	// First line Second column
	#pragma unroll 1
	for(int YYY=1; YYY<scanY; YYY++)
	{
		__syncthreads();
		
		unsigned int x = threadIdx.x +       blockIdx.x*(blockDim.x-1) + blockDim.x-1 +job.start.x; // with x overlap, inner point only.
		unsigned int y = threadIdx.y + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y +YYY*4; // inner point only.
		unsigned int z = threadIdx.z + scanZ*blockIdx.z* blockDim.z                   +job.start.z;
		
		REAL stencil_d[3];  REAL buffer_d[3]; 
		REAL stencil_u[3];  REAL buffer_u[3]; 
		REAL stencil_v[3];  REAL buffer_v[3]; 
		REAL stencil_w[3];  REAL buffer_w[3]; 
		REAL stencil_cc[3]; REAL buffer_cc[3];
		
		REAL stencil_Ax[3]; REAL buffer_Ax[3];
		REAL stencil_Ay[3]; REAL buffer_Ay[3];
		REAL stencil_Az[3]; REAL buffer_Az[3];
		
		REAL Ajacobi;
		REAL rhs[6]; for(int ii=0; ii<6; ii++) rhs[ii] = 0.0; // 0 is dummy, 1~5 is real.
		
		//======================================
		// 			for X-direction
		//======================================
		// /*
		{
			{
				int flag1 = (blockDim.x-4 +threadIdx.x) / blockDim.x; // threadIdx.x < 4,  flag = 0; otherwise flag = 1.
				int flag2 = (blockDim.x+12-threadIdx.x) / blockDim.x; // threadIdx.x > 12, flag = 0; otherwise flag = 1.
				
				unsigned int offset = 1 +threadIdx.x+16*threadIdx.y +64*threadIdx.z;
				
				//
				//for rho parameters
				stencil_d[0] = get_Field_LAP(d, x-4, y, z) ;
				stencil_d[1] = get_Field_LAP(d, x+3, y, z) ;
				
					//store inner flow data to lds buffer
					Trans_shared[ (offset -4) *flag1] = stencil_d[0];
					Trans_shared[ (offset +3) *flag2] = stencil_d[1];
				
				//
				//for u parameters
				stencil_u[0] = get_Field_LAP(u, x-4, y, z) ;
				stencil_u[1] = get_Field_LAP(u, x+3, y, z) ;
					
					//store inner flow data to lds buffer 
					Trans_shared[ ( 256 +offset -4) *flag1] = stencil_u[0];
					Trans_shared[ ( 256 +offset +3) *flag2] = stencil_u[1];
				
				//
				//for v parameters
				stencil_v[0] = get_Field_LAP(v, x-4, y, z) ;
				stencil_v[1] = get_Field_LAP(v, x+3, y, z) ;
				
					//store inner flow data to lds buffer 
					Trans_shared[ ( 256*2 +offset -4) *flag1] = stencil_v[0];
					Trans_shared[ ( 256*2 +offset +3) *flag2] = stencil_v[1];
				
				//
				//for w parameters
				stencil_w[0] = get_Field_LAP(w, x-4, y, z) ;
				stencil_w[1] = get_Field_LAP(w, x+3, y, z) ;
					
					//store inner flow data to lds buffer 
					Trans_shared[ ( 256*3 +offset -4) *flag1] = stencil_w[0];
					Trans_shared[ ( 256*3 +offset +3) *flag2] = stencil_w[1];
				
				//
				//for T/cc parameters
				stencil_cc[0] = get_Field_LAP(cc, x-4, y, z);
				stencil_cc[1] = get_Field_LAP(cc, x+3, y, z);
					
					//store inner flow data to lds buffer 
					Trans_shared[ ( 256*4 +offset -4) *flag1] = stencil_cc[0];
					Trans_shared[ ( 256*4 +offset +3) *flag2] = stencil_cc[1];
				
				//
				//for Ax parameters
				stencil_Ax[0] = get_Field_LAP(Ax, x-4, y, z);
				stencil_Ax[1] = get_Field_LAP(Ax, x+3, y, z);
				
				//
				//for Ay parameters
				stencil_Ay[0] = get_Field_LAP(Ay, x-4, y, z);
				stencil_Ay[1] = get_Field_LAP(Ay, x+3, y, z);
				
				//
				//for Az parameters
				stencil_Az[0] = get_Field_LAP(Az, x-4, y, z);
				stencil_Az[1] = get_Field_LAP(Az, x+3, y, z);
				
				//
				//for jacobian parameters
				Ajacobi = get_Field_LAP(Ajac, x, y, z);
			}
			
			// #pragma unroll 1
			// for(int aaa=0; aaa<1000; aaa++)
			{
				// int offset = threadIdx.x+24*threadIdx.y+96*threadIdx.z; // take x as the continuous direction in LDS. x-y-z: 24*4*4 = 384.
				unsigned int offset = threadIdx.x+23*threadIdx.y+92*threadIdx.z; // take x as the continuous direction in LDS. x-y-z: 23*4*4 = 368.
				// REAL ss[3];
				// REAL E1P[3];
				// REAL E2P[3];
				// REAL E3P[3];
				
				#pragma unroll 1
				for(int ii=0; ii<2; ii++)
				{
					//-------------		
					// x-dir
					//-------------
					REAL ss, Ak1, Ak2, Ak3;
					REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
					REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
					
					ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]); // 5
					REAL rcp = 1 / ss;
					Ak1 = stencil_Ax[ii] *rcp;
					Ak2 = stencil_Ay[ii] *rcp;
					Ak3 = stencil_Az[ii] *rcp; // 10*3

					vs = stencil_Ax[ii] * stencil_u[ii] 
					   + stencil_Ay[ii] * stencil_v[ii]
					   + stencil_Az[ii] * stencil_w[ii]; // 5

					E1 = vs;
					E2 = vs - stencil_cc[ii] * ss;
					E3 = vs + stencil_cc[ii] * ss; // 4

					E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
					E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
					E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

					E1M = E1 - E1P;
					E2M = E2 - E2P;
					E3M = E3 - E3P; // 3
					// ----------------------------------------
					tmp0 = stencil_d[ii] *Gamma_d_rcp; // 1
					uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
					uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
					vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
					vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
					wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
					wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3; // 12
					vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
					vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50; // 12
					vv = (Gamma_d - 1.0) * (stencil_u[ii] * stencil_u[ii] 
										  + stencil_v[ii] * stencil_v[ii]
										  + stencil_w[ii] * stencil_w[ii] ); // 7
					W2 = split_C3_d * stencil_cc[ii] * stencil_cc[ii]; // 2
					
					P_shared[     7*ii +offset] = tmp0 * (split_C1_d * E1P + E2P + E3P); // 1
					M_shared[     7*ii +offset] = tmp0 * (split_C1_d * E1M + E2M + E3M);
					P_shared[ 368+7*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_u[ii] + E2P * uc1 + E3P * uc2); // 2
					M_shared[ 368+7*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_u[ii] + E2M * uc1 + E3M * uc2);
					P_shared[ 736+7*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_v[ii] + E2P * vc1 + E3P * vc2); // 3
					M_shared[ 736+7*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_v[ii] + E2M * vc1 + E3M * vc2);
					P_shared[1104+7*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_w[ii] + E2P * wc1 + E3P * wc2); // 4
					M_shared[1104+7*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_w[ii] + E2M * wc1 + E3M * wc2);
					P_shared[1472+7*ii +offset] = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P)); // 5
					M_shared[1472+7*ii +offset] = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
				}
				
				// __syncthreads();
				REAL weno_P[8], weno_M[8];
				// #pragma unroll 1
				for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[offset +ii],
										  weno_M[ii] = M_shared[offset +ii];
				
				#pragma unroll 1
				for(int Loop=1; Loop<6; Loop++)
				{
					{
						// x_rhs1+
						REAL tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&weno_P[0]);
						REAL tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
						rhs[Loop] += -Ajacobi*(tmp_r - tmp_l)*hx_d_rcp; // put_du_p_kernel
					}
					for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[368*Loop +offset +ii];
					
					{
						// x_rhs1-
						REAL tmp_l = OCFD_weno7_SYMBO_kernel_M_opt (&weno_M[0]);
						REAL tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
						rhs[Loop] += -Ajacobi*(tmp_r - tmp_l)*hx_d_rcp; // put_du_m_kernel
					}
					for(int ii=0; ii<8; ii++) weno_M[ii] = M_shared[368*Loop +offset +ii];
					
					/*
					if(Loop == 2)
					{
						unsigned int offset2 = 1+threadIdx.x+16*threadIdx.y+64*threadIdx.z;
						
						//for rho parameters
						stencil_d[0] = get_Field_LAP(d, x, y-4, z);
						stencil_d[1] = Trans_shared[offset2];
						stencil_d[2] = get_Field_LAP(d, x, y+4, z);

						//for u parameters
						stencil_u[0] = get_Field_LAP(u, x, y-4, z);
						stencil_u[1] = Trans_shared[offset2 +256];
						stencil_u[2] = get_Field_LAP(u, x, y+4, z);
						
						//for v parameters
						stencil_v[0] = get_Field_LAP(v, x, y-4, z);
						stencil_v[1] = Trans_shared[offset2 +256*2];
						stencil_v[2] = get_Field_LAP(v, x, y+4, z);
						
						//for w parameters
						stencil_w[0] = get_Field_LAP(w, x, y-4, z);
						stencil_w[1] = Trans_shared[offset2 +256*3];
						stencil_w[2] = get_Field_LAP(w, x, y+4, z);
						
						//for T/cc parameters
						stencil_cc[0] = get_Field_LAP(cc, x, y-4, z);
						stencil_cc[1] = Trans_shared[offset2 +256*4];
						stencil_cc[2] = get_Field_LAP(cc, x, y+4, z);
						
						//for Ax parameters
						stencil_Ax[0] = get_Field_LAP(Bx, x, y-4, z);
						stencil_Ax[1] = get_Field_LAP(Bx, x, y,   z);
						stencil_Ax[2] = get_Field_LAP(Bx, x, y+4, z);
						
						//for Ay parameters
						stencil_Ay[0] = get_Field_LAP(By, x, y-4, z);
						stencil_Ay[1] = get_Field_LAP(By, x, y,   z);
						stencil_Ay[2] = get_Field_LAP(By, x, y+4, z);
						
						//for Az parameters
						stencil_Az[0] = get_Field_LAP(Bz, x, y-4, z);
						stencil_Az[1] = get_Field_LAP(Bz, x, y,   z);
						stencil_Az[2] = get_Field_LAP(Bz, x, y+4, z);
						// __threadfence();
					}	
					*/
				}
			}
		}

	// #pragma unroll 1
	// for(int TEST=0; TEST<1000; TEST++)
	// {
		// __syncthreads();
		// /*
		//
		//======================================
		// 			for Y-direction
		//======================================
		{
			REAL fpY_buffer[10], fmY_buffer[10];
			REAL stencil_d, stencil_u, stencil_v, stencil_w, stencil_cc;
			REAL stencil_Ax, stencil_Ay, stencil_Az;
			unsigned int offset = threadIdx.y+29*threadIdx.x+464*threadIdx.z; // take y as the continuous direction in LDS. y-z-x: 28*4*16.
			
			__syncthreads();
			{
				fpY_buffer[0] = get_SoA_LAP(fpY, x, y-4, z, 0); // left
				fpY_buffer[1] = get_SoA_LAP(fpY, x, y-4, z, 1);
				fpY_buffer[2] = get_SoA_LAP(fpY, x, y-4, z, 2);
				fpY_buffer[3] = get_SoA_LAP(fpY, x, y-4, z, 3);
				fpY_buffer[4] = get_SoA_LAP(fpY, x, y-4, z, 4);
				fpY_buffer[5] = get_SoA_LAP(fpY, x, y  , z, 0); // center
				fpY_buffer[6] = get_SoA_LAP(fpY, x, y  , z, 1);
				fpY_buffer[7] = get_SoA_LAP(fpY, x, y  , z, 2);
				fpY_buffer[8] = get_SoA_LAP(fpY, x, y  , z, 3);
				fpY_buffer[9] = get_SoA_LAP(fpY, x, y  , z, 4);
				
				P_shared[  offset] = fpY_buffer[0]; // left
				P_shared[4+offset] = fpY_buffer[5]; // center

				fmY_buffer[0] = get_SoA_LAP(fmY, x, y-4, z, 0); // left
				fmY_buffer[1] = get_SoA_LAP(fmY, x, y-4, z, 1);
				fmY_buffer[2] = get_SoA_LAP(fmY, x, y-4, z, 2);
				fmY_buffer[3] = get_SoA_LAP(fmY, x, y-4, z, 3);
				fmY_buffer[4] = get_SoA_LAP(fmY, x, y-4, z, 4);
				fmY_buffer[5] = get_SoA_LAP(fmY, x, y  , z, 0); // center
				fmY_buffer[6] = get_SoA_LAP(fmY, x, y  , z, 1);
				fmY_buffer[7] = get_SoA_LAP(fmY, x, y  , z, 2);
				fmY_buffer[8] = get_SoA_LAP(fmY, x, y  , z, 3);
				fmY_buffer[9] = get_SoA_LAP(fmY, x, y  , z, 4);
				
				M_shared[  offset] = fmY_buffer[0]; // left
				M_shared[4+offset] = fmY_buffer[5]; // center
				
				stencil_d  = get_Field_LAP(d,  x, y+4, z); // right
				stencil_u  = get_Field_LAP(u,  x, y+4, z);
				stencil_v  = get_Field_LAP(v,  x, y+4, z);
				stencil_w  = get_Field_LAP(w,  x, y+4, z);
				stencil_cc = get_Field_LAP(cc, x, y+4, z);
				
				stencil_Ax = get_Field_LAP(Bx, x, y+4, z);
				stencil_Ay = get_Field_LAP(By, x, y+4, z);
				stencil_Az = get_Field_LAP(Bz, x, y+4, z);
			}
			
			
			// if( threadIdx.x != (blockDim.x-1) )
			// {
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = x;
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = y;
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = z;
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = stencil_w[1];
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = stencil_cc[1];
			// }
			
			// if( x>=job.start.x && x<=job.end.x && y>=job.start.y && y<=job.end.y && z>=job.start.z && z<=job.end.z)
			// if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-5)
			// /*
			// #pragma unroll 1
			// for(int aaa=0; aaa<1000; aaa++)
			{
				// y = threadIdx.y + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y +YYY*4;
				// z = threadIdx.z + scanZ*blockIdx.z* blockDim.z                   +job.start.z;
				
				REAL ss;
				REAL E1P;
				REAL E2P;
				REAL E3P;
				REAL weno_P[8], weno_M[8];
				
				// __syncthreads();
				{
					//-------------		
					// y-dir
					//-------------
					REAL Ak1, Ak2, Ak3;
					REAL vs, E1, E2, E3, E1M, E2M, E3M;
					REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
					
					ss = sqrt(stencil_Ax*stencil_Ax + stencil_Ay*stencil_Ay + stencil_Az*stencil_Az); // 5
					REAL rcp = 1 / ss;
					Ak1 = stencil_Ax *rcp;
					Ak2 = stencil_Ay *rcp;
					Ak3 = stencil_Az *rcp; // 10*3

					vs = stencil_Ax * stencil_u 
					   + stencil_Ay * stencil_v
					   + stencil_Az * stencil_w; // 5

					E1 = vs;
					E2 = vs - stencil_cc * ss;
					E3 = vs + stencil_cc * ss; // 4

					E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
					E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
					E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

					E1M = E1 - E1P;
					E2M = E2 - E2P;
					E3M = E3 - E3P; // 3
					// ----------------------------------------
					tmp0 = stencil_d *Gamma_d_rcp; // 1
					uc1  = stencil_u - stencil_cc * Ak1; // 2
					uc2  = stencil_u + stencil_cc * Ak1; // 2
					
					REAL fp1,fm1,fp2,fm2;
					
					fp1 = tmp0 * (split_C1_d * E1P + E2P + E3P); // fpfm_1
					fm1 = tmp0 * (split_C1_d * E1M + E2M + E3M);
					fp2 = tmp0 * (split_C1_d * E1P * stencil_u + E2P * uc1 + E3P * uc2); // fpfm_2
					fm2 = tmp0 * (split_C1_d * E1M * stencil_u + E2M * uc1 + E3M * uc2);
					
					P_shared[8  +offset] = fp1; // fpfm_1
					M_shared[8  +offset] = fm1;
					P_shared[12 +offset] = fp2; // fpfm_2
					M_shared[12 +offset] = fm2;
					
					get_SoA_LAP(fpY, x, y+4, z, 0) = fp1; // fpfm_1
					get_SoA_LAP(fmY, x, y+4, z, 0) = fm1;
					get_SoA_LAP(fpY, x, y+4, z, 1) = fp2; // fpfm_2
					get_SoA_LAP(fmY, x, y+4, z, 1) = fm2;
					
					
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
								// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, tmp0 * (split_C1_d * E1M * stencil_u + E2M * uc1 + E3M * uc2) ); // fpY_buffer[0]
				}
				
						__syncthreads();
						for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[1+offset +ii],
												  weno_M[ii] = M_shared[1+offset +ii];
				
				{
					//-------------		
					// y-dir
					//-------------
					REAL Ak1, Ak2, Ak3;
					REAL vs, E1, E2, E3, E1M, E2M, E3M;
					REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
					
					// ss = sqrt(stencil_Ax*stencil_Ax + stencil_Ay*stencil_Ay + stencil_Az*stencil_Az); // 5
					REAL rcp = 1 / ss;
					Ak1 = stencil_Ax *rcp;
					Ak2 = stencil_Ay *rcp;
					Ak3 = stencil_Az *rcp; // 10*3

					vs = stencil_Ax * stencil_u 
					   + stencil_Ay * stencil_v
					   + stencil_Az * stencil_w; // 5

					E1 = vs;
					E2 = vs - stencil_cc * ss;
					E3 = vs + stencil_cc * ss; // 4

					// E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
					// E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
					// E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

					E1M = E1 - E1P;
					E2M = E2 - E2P;
					E3M = E3 - E3P; // 3
					// ----------------------------------------
					tmp0 = stencil_d *Gamma_d_rcp; // 1
					uc1  = stencil_u - stencil_cc * Ak1;
					uc2  = stencil_u + stencil_cc * Ak1;
					vc1  = stencil_v - stencil_cc * Ak2;
					vc2  = stencil_v + stencil_cc * Ak2;
					wc1  = stencil_w - stencil_cc * Ak3;
					wc2  = stencil_w + stencil_cc * Ak3; // 12
					vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
					vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50; // 12
					vv = (Gamma_d - 1.0) * (stencil_u * stencil_u 
										  + stencil_v * stencil_v
										  + stencil_w * stencil_w ); // 7
					W2 = split_C3_d * stencil_cc * stencil_cc; // 2
					
					// P_shared[8  +offset] = tmp0 * (split_C1_d * E1P + E2P + E3P); // fpfm_1
					// M_shared[8  +offset] = tmp0 * (split_C1_d * E1M + E2M + E3M);
					// P_shared[12 +offset] = tmp0 * (split_C1_d * E1P * stencil_u + E2P * uc1 + E3P * uc2); // fpfm_2
					// M_shared[12 +offset] = tmp0 * (split_C1_d * E1M * stencil_u + E2M * uc1 + E3M * uc2);
					
					REAL fp3,fm3,fp4,fm4,fp5,fm5;
					
					fp3 = tmp0 * (split_C1_d * E1P * stencil_v + E2P * vc1 + E3P * vc2); // fpfm_3
					fm3 = tmp0 * (split_C1_d * E1M * stencil_v + E2M * vc1 + E3M * vc2);
					fp4 = tmp0 * (split_C1_d * E1P * stencil_w + E2P * wc1 + E3P * wc2); // fpfm_4
					fm4 = tmp0 * (split_C1_d * E1M * stencil_w + E2M * wc1 + E3M * wc2);
					fp5 = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P)); // fpfm_5
					fm5 = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
					
					P_shared[16 +offset] = fp3; // fpfm_3
					M_shared[16 +offset] = fm3;
					P_shared[20 +offset] = fp4; // fpfm_4
					M_shared[20 +offset] = fm4;
					P_shared[24 +offset] = fp5; // fpfm_5
					M_shared[24 +offset] = fm5;
					
					get_SoA_LAP(fpY, x, y+4, z, 2) = fp3; // fpfm_3
					get_SoA_LAP(fmY, x, y+4, z, 2) = fm3;
					get_SoA_LAP(fpY, x, y+4, z, 3) = fp4; // fpfm_4
					get_SoA_LAP(fmY, x, y+4, z, 3) = fm4;
					get_SoA_LAP(fpY, x, y+4, z, 4) = fp5; // fpfm_5
					get_SoA_LAP(fmY, x, y+4, z, 4) = fm5;
				}
				
						// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
							// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, P_shared[12 +offset]);
				
						// __syncthreads();
						P_shared[4+offset] = fpY_buffer[1]; // fp_2 Left
						P_shared[8+offset] = fpY_buffer[6]; // fp_2 center
				
				#pragma unroll 1
				for(int loop=1; loop<6; loop++)
				{
					int flag_faceY = !threadIdx.y; // 1,0,0,0
					REAL weno_plus;
					
					{
						// weno+;
						REAL weno = OCFD_weno7_SYMBO_kernel_P_opt(&weno_P[0]);
						REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
						rhs[loop] += -Ajacobi*( weno - UP*!flag_faceY )*hy_d_rcp;
						weno_plus = weno;
						
								// if(ZZZ == 1 && loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x==0 && threadIdx.y==1 && threadIdx.z==0)
									// printf("YYY=%d, weno_P: %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E\n", YYY, weno_P[0],weno_P[1],weno_P[2],weno_P[3],weno_P[4],weno_P[5],weno_P[6],weno_P[7]);
								
								// if(ZZZ == 1 && loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
									// printf("YYY=%d, threadIdx.x=%d && threadIdx.y=%d && threadIdx.z=%d,\n weno_P: %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E\n", YYY, threadIdx.x,threadIdx.y,threadIdx.z, 
									// weno_P[0],weno_P[1],weno_P[2],weno_P[3],weno_P[4],weno_P[5],weno_P[6],weno_P[7]);
									
								// if(ZZZ == 1 && loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
									// printf("YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", YYY, threadIdx.x,threadIdx.y,threadIdx.z, weno);
								
								// if(loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
									// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, fpY_buffer[0]); // fpY_buffer[0]
					}
					
							if(loop<5)
							{
								M_shared[  loop*4+offset] = fmY_buffer[loop], 
								M_shared[4+loop*4+offset] = fmY_buffer[loop+5]; // WRITE: fm_2345*(左中)
								for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[1+loop*4+offset +ii]; // READ:  fp_2345*(左中右)
							}
				
					int offset_Previous = -63+loop*64 +threadIdx.x+16*threadIdx.z; // take x as the continuous direction in LDS. x-z: 16*4.
					REAL weno_Previous = FaceY_shared[offset_Previous];
					
					{
						// weno-;
						REAL weno = OCFD_weno7_SYMBO_kernel_M_opt(&weno_M[0]);
						REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
						rhs[loop] += -Ajacobi*( weno - UP*!flag_faceY )*hy_d_rcp;
						int flag_thread = threadIdx.y / 3; // 0,0,0,1
						FaceY_shared[offset_Previous *flag_thread] = weno_plus + weno;
						
								// if(ZZZ == 1 && loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
									// printf("YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", YYY, threadIdx.x,threadIdx.y,threadIdx.z, weno);
								// if(loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
									// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, weno);
								// if(loop == 1)
									// printf("ZZZ=%d, YYY=%d; blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, blockIdx.x,blockIdx.y,blockIdx.z, threadIdx.x,threadIdx.y,threadIdx.z, weno);
								
								// if(loop == 5 && blockIdx.x==1 && blockIdx.y==1 && blockIdx.z==1)
									// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, weno);
					}
					rhs[loop] += Ajacobi *weno_Previous *hy_d_rcp *flag_faceY;
					
							int flag_loop = loop/4; // 0,0,0,1,1
							if(loop<5)
							{
								P_shared[ (4+loop*4+offset)* !flag_loop] = fpY_buffer[loop+1], 
								P_shared[ (8+loop*4+offset)* !flag_loop] = fpY_buffer[loop+6]; // WRITE: fp_345**(左中)
								for(int ii=0; ii<8; ii++) weno_M[ii] = M_shared[1+loop*4+offset +ii]; // READ:  fm_2345*(左中右)
							}
						
						
								// if(loop == 4 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
									// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, rhs[loop] );
				}
			}
			// */
		}
		// */
		
		
		//======================================
		// 			for Z-direction
		//======================================
		// /*
		{
			__syncthreads();
			// y = threadIdx.z + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y +YYY*4;
			// z = threadIdx.y + scanZ*blockIdx.z* blockDim.z                   +job.start.z;
			
			{
				y = threadIdx.z + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y +YYY*4;
				z = threadIdx.y + scanZ*blockIdx.z* blockDim.z                   +job.start.z;
				unsigned int offset2 = 1+threadIdx.x+16*threadIdx.z+64*threadIdx.y; // 5
		
				//for rho parameters
				buffer_d[0] = get_Field_LAP(d, x, y, z-4); // 6+1
				buffer_d[1] = Trans_shared[offset2];
				buffer_d[2] = get_Field_LAP(d, x, y, z+4);
				// buffer_d[2] = get_Field_LAP(d, x, y, z+4-flag*8);
				
				//for u parameters
				buffer_u[0] = get_Field_LAP(u, x, y, z-4) ;
				buffer_u[1] = Trans_shared[offset2 +256];
				buffer_u[2] = get_Field_LAP(u, x, y, z+4);
				
				//for v parameters
				buffer_v[0] = get_Field_LAP(v, x, y, z-4) ;
				buffer_v[1] = Trans_shared[offset2 +256*2];
				buffer_v[2] = get_Field_LAP(v, x, y, z+4);
			
				//for w parameters
				buffer_w[0] = get_Field_LAP(w, x, y, z-4) ;
				buffer_w[1] = Trans_shared[offset2 +256*3];
				buffer_w[2] = get_Field_LAP(w, x, y, z+4);
				
				//for T/cc parameters
				buffer_cc[0] = get_Field_LAP(cc, x, y, z-4);
				buffer_cc[1] = Trans_shared[offset2 +256*4];
				buffer_cc[2] = get_Field_LAP(cc, x, y, z+4);
				
				//for Ax parameters
				buffer_Ax[0] = get_Field_LAP(Cx, x, y, z-4) ;
				buffer_Ax[1] = get_Field_LAP(Cx, x, y, z  ) ;
				buffer_Ax[2] = get_Field_LAP(Cx, x, y, z+4);
				
				//for Ay parameters
				buffer_Ay[0] = get_Field_LAP(Cy, x, y, z-4) ;
				buffer_Ay[1] = get_Field_LAP(Cy, x, y, z  ) ;
				buffer_Ay[2] = get_Field_LAP(Cy, x, y, z+4);
				
				//for Az parameters
				buffer_Az[0] = get_Field_LAP(Cz, x, y, z-4) ;
				buffer_Az[1] = get_Field_LAP(Cz, x, y, z  ) ;
				buffer_Az[2] = get_Field_LAP(Cz, x, y, z+4);
			}
			
			REAL RHS[6]; for(int ii=0; ii<6; ii++) RHS[ii] = 0.0;
			{
				unsigned int offset_tran = threadIdx.z+4*threadIdx.y+16*threadIdx.x;
				Trans_shared[      offset_tran] = rhs[1];
				Trans_shared[256  +offset_tran] = rhs[2];
				Trans_shared[512  +offset_tran] = rhs[3];
				Trans_shared[768  +offset_tran] = rhs[4];
				Trans_shared[1024 +offset_tran] = rhs[5];
				
				Ajacobi = get_Field_LAP(Ajac, x, y, z);
				// __syncthreads();
				// __threadfence();
					// __asm__(
					// " s_waitcnt lgkmcnt(0)  \n\t"
					// " s_waitcnt vmcnt(0)  \n\t"
					// );
			}
			
			// #pragma unroll 1
			// for(int aaa=0; aaa<1000; aaa++)
			{
				unsigned int offset = threadIdx.y+29*threadIdx.x+464*threadIdx.z; // take y as the continuous direction in LDS. y-z-x: 28*4*16.
				REAL ss[3];
				REAL E1P[3];
				REAL E2P[3];
				REAL E3P[3];
				REAL weno_P[8], weno_M[8];
				
				#pragma unroll 1
				for(int ii=0; ii<3; ii++)
				{
					//-------------		
					// z-dir-fpfm-5
					//-------------
					REAL Ak1, Ak2, Ak3;
					REAL vs, E1, E2, E3;
					// REAL E1P, E2P, E3P;
					REAL E1M, E2M, E3M;
					REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
					
					ss[ii] = sqrt(buffer_Ax[ii]*buffer_Ax[ii] + buffer_Ay[ii]*buffer_Ay[ii] + buffer_Az[ii]*buffer_Az[ii]); // 5
					
					REAL rcp = 1 / ss[ii];
					Ak1 = buffer_Ax[ii] *rcp;
					Ak2 = buffer_Ay[ii] *rcp;
					Ak3 = buffer_Az[ii] *rcp; // 10*3

					vs = buffer_Ax[ii] * buffer_u[ii] 
					   + buffer_Ay[ii] * buffer_v[ii]
					   + buffer_Az[ii] * buffer_w[ii]; // 5

					E1 = vs;
					E2 = vs - buffer_cc[ii] * ss[ii];
					E3 = vs + buffer_cc[ii] * ss[ii]; // 4

					E1P[ii] = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
					E2P[ii] = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
					E3P[ii] = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

					E1M = E1 - E1P[ii];
					E2M = E2 - E2P[ii];
					E3M = E3 - E3P[ii]; // 3
					// ----------------------------------------
					tmp0 = buffer_d[ii] *Gamma_d_rcp; // 1
					uc1  = buffer_u[ii] - buffer_cc[ii] * Ak1;
					uc2  = buffer_u[ii] + buffer_cc[ii] * Ak1;
					vc1  = buffer_v[ii] - buffer_cc[ii] * Ak2;
					vc2  = buffer_v[ii] + buffer_cc[ii] * Ak2;
					wc1  = buffer_w[ii] - buffer_cc[ii] * Ak3;
					wc2  = buffer_w[ii] + buffer_cc[ii] * Ak3; // 12
					vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
					vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50; // 12
					vv = (Gamma_d - 1.0) * (buffer_u[ii] * buffer_u[ii] 
										  + buffer_v[ii] * buffer_v[ii]
										  + buffer_w[ii] * buffer_w[ii] ); // 7
					W2 = split_C3_d * buffer_cc[ii] * buffer_cc[ii]; // 2
					
					// take z as the continuous direction in LDS. z-y-x: 28*4*16
					REAL fp5, fm5;
					fp5 = tmp0 * (E1P[ii] * vv + E2P[ii] * vvc1 + E3P[ii] * vvc2 + W2 * (E2P[ii] + E3P[ii]));
					fm5 = tmp0 * (E1M     * vv + E2M     * vvc1 + E3M     * vvc2 + W2 * (E2M     + E3M)); // 18
					
					P_shared[4*ii +offset] = fp5;
					M_shared[4*ii +offset] = fm5;
					
					get_SoA_LAP(fpZ, x, y, z+4*ii-4, 4) = fp5; // fpfm_5
					get_SoA_LAP(fmZ, x, y, z+4*ii-4, 4) = fm5;
				}
				
				// z-dir-fpfm-1
				// for(int ii=0; ii<3; ii++)
				{
					//-------------		
					// z-dir-fpfm-1
					//-------------
					const int ii = 0;
					// REAL ss;
					REAL vs, E1, E2, E3;
					// REAL E1P, E2P, E3P;
					REAL E1M, E2M, E3M;
					REAL tmp0;
					
					// ss[ii] = sqrt(buffer_Ax[ii]*buffer_Ax[ii] + buffer_Ay[ii]*buffer_Ay[ii] + buffer_Az[ii]*buffer_Az[ii]); // 5

					vs = buffer_Ax[ii] * buffer_u[ii] 
					   + buffer_Ay[ii] * buffer_v[ii]
					   + buffer_Az[ii] * buffer_w[ii]; // 5

					E1 = vs;
					E2 = vs - buffer_cc[ii] * ss[ii];
					E3 = vs + buffer_cc[ii] * ss[ii]; // 4

					// E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
					// E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
					// E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

					E1M = E1 - E1P[ii];
					E2M = E2 - E2P[ii];
					E3M = E3 - E3P[ii]; // 3
					// ----------------------------------------
					tmp0 = buffer_d[ii] *Gamma_d_rcp; // 1
					
					REAL fp1, fm1;
					fp1 = tmp0 * (split_C1_d * E1P[ii] + E2P[ii] + E3P[ii]);
					fm1 = tmp0 * (split_C1_d * E1M     + E2M +     E3M); // 8
					
					P_shared[12 +4*ii +offset] = fp1;
					M_shared[12 +4*ii +offset] = fm1;
					
					get_SoA_LAP(fpZ, x, y, z-4, 0) = fp1; // fpfm_1
					get_SoA_LAP(fmZ, x, y, z-4, 0) = fm1;
				}
				
				__syncthreads();
								///////////////////////////
								// buffer_1 prefetching: y_Loop1+: rhs1+
								// #pragma unroll 1
								for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[offset +ii],
														  weno_M[ii] = M_shared[offset +ii];
								
								// for(int ii=4; ii<8; ii++) weno_P[ii] = P_shared[offset +ii],
														  // weno_M[ii] = M_shared[offset +ii];
														  
								// unsigned int offset_tran = threadIdx.y+4*threadIdx.z+16*threadIdx.x;
								// rhs[1] = Trans_shared[      offset_tran] ;
								// rhs[2] = Trans_shared[256  +offset_tran] ;
								// rhs[3] = Trans_shared[512  +offset_tran] ;
								// rhs[4] = Trans_shared[768  +offset_tran] ;
								// rhs[5] = Trans_shared[1024 +offset_tran] ;
								
				
				// __syncthreads();
				// __threadfence();
				offset -= threadIdx.y;
				REAL Wp_previous = 0.0, Wm_previous = 0.0;
				int point = 1;
				
				#pragma unroll 1
				for(int loop=0; loop<7; loop++)
				{
					// if(loop == 0)
					// {
						// for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[offset +threadIdx.y +ii],
												  // weno_M[ii] = M_shared[offset +threadIdx.y +ii];
						
						// unsigned int offset_tran = threadIdx.y+4*threadIdx.z+16*threadIdx.x;
						// rhs[1] = Trans_shared[256  +offset_tran] ;
						// rhs[2] = Trans_shared[512  +offset_tran] ;
						// rhs[3] = Trans_shared[768  +offset_tran] ;
						// rhs[4] = Trans_shared[1024 +offset_tran] ;
						// rhs[5] = Trans_shared[1280 +offset_tran] ;
					// }
					
					#pragma unroll 1
					for(int nn=0; nn<2; nn++)
					{
						// if(loop == 6) break;
						if(loop+nn >= 6) break;
						
						//-------------
						// z-dir-1234
						//-------------
						
						int u3Bits = ( 0b110110110010001000100011010001110000101100 >> 3*(loop*2+nn) ) & 0b111;
						
						// int u2Bits_2 = ( 0b1001001001001001001001 >> 2*(loop*2+nn) ) & 0b11;
						
						// int u2Bits_1 = ( 0b1111111010100101010000 >> 2*(loop*2+nn) ) & 0b11;
						
						// int ii = u2Bits_2;
						int ii = point % 3;
						
						REAL Ak1, Ak2, Ak3;
						REAL vs, E1, E2, E3;
						// REAL E1P, E2P, E3P;
						REAL E1M, E2M, E3M;
						REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2;
						
						// ss[ii] = sqrt(buffer_Ax[ii]*buffer_Ax[ii] + buffer_Ay[ii]*buffer_Ay[ii] + buffer_Az[ii]*buffer_Az[ii]); // 5
						REAL rcp = 1 / ss[ii];
						Ak1 = buffer_Ax[ii] *rcp;
						Ak2 = buffer_Ay[ii] *rcp;
						Ak3 = buffer_Az[ii] *rcp; // 10*3

						vs = buffer_Ax[ii] * buffer_u[ii] 
						   + buffer_Ay[ii] * buffer_v[ii]
						   + buffer_Az[ii] * buffer_w[ii]; // 5

						E1 = vs;
						E2 = vs - buffer_cc[ii] * ss[ii];
						E3 = vs + buffer_cc[ii] * ss[ii]; // 4

						// E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 5*3

						E1M = E1 - E1P[ii];
						E2M = E2 - E2P[ii];
						E3M = E3 - E3P[ii]; // 3
						// ----------------------------------------
						tmp0 = buffer_d[ii] *Gamma_d_rcp;
						uc1  = buffer_u[ii] - buffer_cc[ii] * Ak1;
						uc2  = buffer_u[ii] + buffer_cc[ii] * Ak1;
						vc1  = buffer_v[ii] - buffer_cc[ii] * Ak2;
						vc2  = buffer_v[ii] + buffer_cc[ii] * Ak2;
						wc1  = buffer_w[ii] - buffer_cc[ii] * Ak3;
						wc2  = buffer_w[ii] + buffer_cc[ii] * Ak3; // 6*2+1
						
						double N1[4] = {1.0, buffer_u[ii], buffer_v[ii], buffer_w[ii]};
						double N2[4] = {1.0, uc1, vc1, wc1};
						double N3[4] = {1.0, uc2, vc2, wc2};
						
						// int fml = u2Bits_1; // formula
						int fml = point / 3; // formula
						point += 1;
						
						REAL fp, fm;
						fp = tmp0 * ( N1[fml]*split_C1_d*E1P[ii]  +  N2[fml]*E2P[ii]  +  N3[fml]*E3P[ii] );
						fm = tmp0 * ( N1[fml]*split_C1_d*E1M      +  N2[fml]*E2M      +  N3[fml]*E3M );// 7*2
						
						P_shared[4*u3Bits +threadIdx.y+offset] = fp;
						M_shared[4*u3Bits +threadIdx.y+offset] = fm;
						
						get_SoA_LAP(fpZ, x, y, z+4*ii-4, fml) = fp;
						get_SoA_LAP(fmZ, x, y, z+4*ii-4, fml) = fm;
					}
					
					// __syncthreads();
					uint32_t rhs_flag[4] = {0b1001000001001001000001000100, 0b1010000001001010100100000100,
											0b1010000001001010101000000100, 0b1010100100001010101010010000};
					int rhs_flag_w = ( rhs_flag[threadIdx.y] >> (loop*4+3) ) & 0b1;
					int rhs_flag_D = ( rhs_flag[threadIdx.y] >> (loop*4+2) ) & 0b1;
					int rhs_flag_U = ( rhs_flag[threadIdx.y] >> (loop*4+1) ) & 0b1;
					int rhs_flag_P = ( rhs_flag[threadIdx.y] >> (loop*4  ) ) & 0b1;
					
					uint32_t rhs_ID[4] = {0b100000011010000001101, 0b100000011010001000101,
										  0b100000011010001000101, 0b100011000010001101000};
					int rhs_id = ( rhs_ID[threadIdx.y] >> (loop*3) ) & 0b111;
					{
						// z_rhs+
						REAL weno = OCFD_weno7_SYMBO_kernel_P_opt(&weno_P[0]);
						REAL DOWN = __shfl_down_double(weno, 16, hipWarpSize);
						REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
						RHS[rhs_id] += -Ajacobi*( (2*rhs_flag_w-1)*weno +rhs_flag_D*DOWN -rhs_flag_U*UP -rhs_flag_P*Wp_previous )*hz_d_rcp; // 10
						Wp_previous = weno;
					}
					
					{
						// z_rhs-
						REAL weno = OCFD_weno7_SYMBO_kernel_M_opt(&weno_M[0]);
						REAL DOWN = __shfl_down_double(weno, 16, hipWarpSize);
						REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
						RHS[rhs_id] += -Ajacobi*( (2*rhs_flag_w-1)*weno +rhs_flag_D*DOWN -rhs_flag_U*UP -rhs_flag_P*Wm_previous )*hz_d_rcp;
						Wm_previous = weno;
					}
					
					int u3Face = ( 0b011010000001000100000 >> 3*loop ) & 0b111; // 0,5,1,2,0,3,4
					int flag_face = ( 0b1101110 >> loop ) & 0b1; // 0,1,1,1,0,1,1
					int flag_thread = threadIdx.y / 3; // 0,0,0,1
					int offset_face = YYY*320 +1+u3Face*64 +threadIdx.x+16*threadIdx.z; // take x as the continuous direction in LDS. x-z: 16*4.
					FaceZ_shared[offset_face *flag_face *flag_thread] = Wp_previous + Wm_previous;
					
					// __syncthreads();
					uint64_t lds_load[7][4] = {
						// {0b0011100110001010010000011000100000100000, 0b0100000111001100010100100000110001000001, 0b0100101000001110011000101001000001100010, 0b0101001001010000011100110001010010000011}, // 1
						{0b1001110010100011000001111011100110101100, 0b1010010011100101000110000011110111001101, 
						 0, 0b0101101010010010100000111001100010100100}, // 1
						{0b1101111010110011100000011000100000100000, 0b1010110100100111001010001100000111101110,
						 0b1011010101101001001110010100011000001111, 0b1011110110101011010010011100101000110000}, // 2
						{0b0010011011110101100111000000110001000001, 0b0010100100110111101011001110000001100010,
						 0b0011000101001001101111010110011100000011, 0b0011100110001010010011011110101100111000}, // 3
						{0b0111101110011010110001011010100100101000, 0b1000001111011100110101100010110101001001,
						 0b1000110000011110111001101011000101101010, 0b1001010001100000111101110011010110001011}, // 4
						{0b0011100110001010010000011000100000100000, 0,
						 0 , 0b1001110010100011000001111011100110101100}, // 5
						{0b0100000111001100010100100000110001000001, 0b0100101000001110011000101001000001100010,
						 0b0101001001010000011100110001010010000011, 0b0101101010010010100000111001100010100100}, // 6
						{0, 0, 0, 0} // 7
					};
					
					// #pragma unroll 1
					for(int ii=0; ii<8; ii++)
					{
						if(loop == 6) break;
						int offset_LDS = ( lds_load[loop][threadIdx.y] >> (ii * 5) ) & 0b11111;
						weno_P[ii] = P_shared[offset_LDS +offset];
						weno_M[ii] = M_shared[offset_LDS +offset];
					}
					
					// __syncthreads();
					
					// if( threadIdx.x != (blockDim.x-1) && loop > 1)
					// {
						// int GLB[5] = {4,0,1,2,3};
						// int num = GLB[loop-2]; // loop-2: 0,1,2,3,4
						// unsigned int offset_tran = threadIdx.y+4*threadIdx.z+16*threadIdx.x;
						// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, num) = RHS[num+1] +Trans_shared[256*num+offset_tran];
					// }
					
					// if(loop == 5)
					// {
						// unsigned int offset_tran = threadIdx.y+4*threadIdx.z+16*threadIdx.x;
						// rhs[1] = Trans_shared[      offset_tran] ;
						// rhs[2] = Trans_shared[256  +offset_tran] ;
						// rhs[3] = Trans_shared[512  +offset_tran] ;
						// rhs[4] = Trans_shared[768  +offset_tran] ;
						// rhs[5] = Trans_shared[1024 +offset_tran] ;
					// }
				}
				
			}
			
			unsigned int offset_tran = threadIdx.y+4*threadIdx.z+16*threadIdx.x;
			if( threadIdx.x != (blockDim.x-1) )
			{
				get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = RHS[1] +Trans_shared[      offset_tran];
				get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = RHS[2] +Trans_shared[256  +offset_tran];
				get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = RHS[3] +Trans_shared[512  +offset_tran];
				get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = RHS[4] +Trans_shared[768  +offset_tran];
				get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = RHS[5] +Trans_shared[1024 +offset_tran];
				
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = RHS[1] +rhs[1];
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = RHS[2] +rhs[2];
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = RHS[3] +rhs[3];
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = RHS[4] +rhs[4];
				// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = RHS[5] +rhs[5];
			}
		}
	}
	
	
	///////////////////////////////////////////////////////////////////////
	///////////////////////////// Second line /////////////////////////////
	#pragma unroll 1
	for(int ZZZ=1; ZZZ<scanZ; ZZZ++)
	{
		// Second line first tile.
		{
			__syncthreads();
			
			unsigned int x = threadIdx.x +       blockIdx.x*(blockDim.x-1) + blockDim.x-1 +job.start.x; // with x overlap, inner point only.
			unsigned int y = threadIdx.y + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y; // inner point only.
			unsigned int z = threadIdx.z + scanZ*blockIdx.z* blockDim.z                   +job.start.z +ZZZ*4;
			
			REAL stencil_d[3];  REAL buffer_d[3]; 
			REAL stencil_u[3];  REAL buffer_u[3]; 
			REAL stencil_v[3];  REAL buffer_v[3]; 
			REAL stencil_w[3];  REAL buffer_w[3]; 
			REAL stencil_cc[3]; REAL buffer_cc[3];
			
			REAL stencil_Ax[3]; REAL buffer_Ax[3];
			REAL stencil_Ay[3]; REAL buffer_Ay[3];
			REAL stencil_Az[3]; REAL buffer_Az[3];
			
			REAL Ajacobi;
			REAL rhs[6]; for(int ii=0; ii<6; ii++) rhs[ii] = 0.0; // 0 is dummy, 1~5 is real.
			
			//======================================
			// 			for X-direction
			//======================================
			// /*
			{
				{
					int flag1 = (blockDim.x-4 +threadIdx.x) / blockDim.x; // threadIdx.x < 4,  flag = 0; otherwise flag = 1.
					int flag2 = (blockDim.x+12-threadIdx.x) / blockDim.x; // threadIdx.x > 12, flag = 0; otherwise flag = 1.
					
					unsigned int offset = 1 +threadIdx.x+16*threadIdx.y +64*threadIdx.z;
					
					//
					//for rho parameters
					stencil_d[0] = get_Field_LAP(d, x-4, y, z) ;
					stencil_d[1] = get_Field_LAP(d, x+3, y, z) ;
					
						//store inner flow data to lds buffer
						Trans_shared[ (offset -4) *flag1] = stencil_d[0];
						Trans_shared[ (offset +3) *flag2] = stencil_d[1];
					
					//
					//for u parameters
					stencil_u[0] = get_Field_LAP(u, x-4, y, z) ;
					stencil_u[1] = get_Field_LAP(u, x+3, y, z) ;
						
						//store inner flow data to lds buffer 
						Trans_shared[ ( 256 +offset -4) *flag1] = stencil_u[0];
						Trans_shared[ ( 256 +offset +3) *flag2] = stencil_u[1];
					
					//
					//for v parameters
					stencil_v[0] = get_Field_LAP(v, x-4, y, z) ;
					stencil_v[1] = get_Field_LAP(v, x+3, y, z) ;
					
						//store inner flow data to lds buffer 
						Trans_shared[ ( 256*2 +offset -4) *flag1] = stencil_v[0];
						Trans_shared[ ( 256*2 +offset +3) *flag2] = stencil_v[1];
					
					//
					//for w parameters
					stencil_w[0] = get_Field_LAP(w, x-4, y, z) ;
					stencil_w[1] = get_Field_LAP(w, x+3, y, z) ;
						
						//store inner flow data to lds buffer 
						Trans_shared[ ( 256*3 +offset -4) *flag1] = stencil_w[0];
						Trans_shared[ ( 256*3 +offset +3) *flag2] = stencil_w[1];
					
					//
					//for T/cc parameters
					stencil_cc[0] = get_Field_LAP(cc, x-4, y, z);
					stencil_cc[1] = get_Field_LAP(cc, x+3, y, z);
						
						//store inner flow data to lds buffer 
						Trans_shared[ ( 256*4 +offset -4) *flag1] = stencil_cc[0];
						Trans_shared[ ( 256*4 +offset +3) *flag2] = stencil_cc[1];
					
					//
					//for Ax parameters
					stencil_Ax[0] = get_Field_LAP(Ax, x-4, y, z);
					stencil_Ax[1] = get_Field_LAP(Ax, x+3, y, z);
					
					//
					//for Ay parameters
					stencil_Ay[0] = get_Field_LAP(Ay, x-4, y, z);
					stencil_Ay[1] = get_Field_LAP(Ay, x+3, y, z);
					
					//
					//for Az parameters
					stencil_Az[0] = get_Field_LAP(Az, x-4, y, z);
					stencil_Az[1] = get_Field_LAP(Az, x+3, y, z);
					
					//
					//for jacobian parameters
					Ajacobi = get_Field_LAP(Ajac, x, y, z);
				}
				
				// #pragma unroll 1
				// for(int aaa=0; aaa<1000; aaa++)
				{
					// int offset = threadIdx.x+24*threadIdx.y+96*threadIdx.z; // take x as the continuous direction in LDS. x-y-z: 24*4*4 = 384.
					unsigned int offset = threadIdx.x+23*threadIdx.y+92*threadIdx.z; // take x as the continuous direction in LDS. x-y-z: 23*4*4 = 368.
					// REAL ss[3];
					// REAL E1P[3];
					// REAL E2P[3];
					// REAL E3P[3];
					
					#pragma unroll 1
					for(int ii=0; ii<2; ii++)
					{
						//-------------		
						// x-dir
						//-------------
						REAL ss, Ak1, Ak2, Ak3;
						REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
						REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
						
						ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]); // 5
						REAL rcp = 1 / ss;
						Ak1 = stencil_Ax[ii] *rcp;
						Ak2 = stencil_Ay[ii] *rcp;
						Ak3 = stencil_Az[ii] *rcp; // 10*3

						vs = stencil_Ax[ii] * stencil_u[ii] 
						   + stencil_Ay[ii] * stencil_v[ii]
						   + stencil_Az[ii] * stencil_w[ii]; // 5

						E1 = vs;
						E2 = vs - stencil_cc[ii] * ss;
						E3 = vs + stencil_cc[ii] * ss; // 4

						E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

						E1M = E1 - E1P;
						E2M = E2 - E2P;
						E3M = E3 - E3P; // 3
						// ----------------------------------------
						tmp0 = stencil_d[ii] *Gamma_d_rcp; // 1
						uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
						uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
						vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
						vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
						wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
						wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3; // 12
						vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
						vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50; // 12
						vv = (Gamma_d - 1.0) * (stencil_u[ii] * stencil_u[ii] 
											  + stencil_v[ii] * stencil_v[ii]
											  + stencil_w[ii] * stencil_w[ii] ); // 7
						W2 = split_C3_d * stencil_cc[ii] * stencil_cc[ii]; // 2
						
						P_shared[     7*ii +offset] = tmp0 * (split_C1_d * E1P + E2P + E3P); // 1
						M_shared[     7*ii +offset] = tmp0 * (split_C1_d * E1M + E2M + E3M);
						P_shared[ 368+7*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_u[ii] + E2P * uc1 + E3P * uc2); // 2
						M_shared[ 368+7*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_u[ii] + E2M * uc1 + E3M * uc2);
						P_shared[ 736+7*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_v[ii] + E2P * vc1 + E3P * vc2); // 3
						M_shared[ 736+7*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_v[ii] + E2M * vc1 + E3M * vc2);
						P_shared[1104+7*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_w[ii] + E2P * wc1 + E3P * wc2); // 4
						M_shared[1104+7*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_w[ii] + E2M * wc1 + E3M * wc2);
						P_shared[1472+7*ii +offset] = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P)); // 5
						M_shared[1472+7*ii +offset] = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
					}
					
					// __threadfence();
					// __syncthreads();
						
					REAL weno_P[8], weno_M[8];
					// #pragma unroll 1
					for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[offset +ii],
											  weno_M[ii] = M_shared[offset +ii];
					
					#pragma unroll 1
					for(int Loop=1; Loop<6; Loop++)
					{
						{
							// x_rhs1+
							REAL tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&weno_P[0]);
							REAL tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
							rhs[Loop] += -Ajacobi*(tmp_r - tmp_l)*hx_d_rcp; // put_du_p_kernel
						}
						for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[368*Loop +offset +ii];
						
						{
							// x_rhs1-
							REAL tmp_l = OCFD_weno7_SYMBO_kernel_M_opt (&weno_M[0]);
							REAL tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
							rhs[Loop] += -Ajacobi*(tmp_r - tmp_l)*hx_d_rcp; // put_du_m_kernel
						}
						for(int ii=0; ii<8; ii++) weno_M[ii] = M_shared[368*Loop +offset +ii];
						
						if(Loop == 2)
						{
							unsigned int offset2 = 1+threadIdx.x+16*threadIdx.y+64*threadIdx.z;
							
							//for rho parameters
							stencil_d[0] = get_Field_LAP(d, x, y-4, z);
							stencil_d[1] = Trans_shared[offset2];
							stencil_d[2] = get_Field_LAP(d, x, y+4, z);

							//for u parameters
							stencil_u[0] = get_Field_LAP(u, x, y-4, z);
							stencil_u[1] = Trans_shared[offset2 +256];
							stencil_u[2] = get_Field_LAP(u, x, y+4, z);
							
							//for v parameters
							stencil_v[0] = get_Field_LAP(v, x, y-4, z);
							stencil_v[1] = Trans_shared[offset2 +256*2];
							stencil_v[2] = get_Field_LAP(v, x, y+4, z);
							
							//for w parameters
							stencil_w[0] = get_Field_LAP(w, x, y-4, z);
							stencil_w[1] = Trans_shared[offset2 +256*3];
							stencil_w[2] = get_Field_LAP(w, x, y+4, z);
							
							//for T/cc parameters
							stencil_cc[0] = get_Field_LAP(cc, x, y-4, z);
							stencil_cc[1] = Trans_shared[offset2 +256*4];
							stencil_cc[2] = get_Field_LAP(cc, x, y+4, z);
							
							//for Ax parameters
							stencil_Ax[0] = get_Field_LAP(Bx, x, y-4, z);
							stencil_Ax[1] = get_Field_LAP(Bx, x, y,   z);
							stencil_Ax[2] = get_Field_LAP(Bx, x, y+4, z);
							
							//for Ay parameters
							stencil_Ay[0] = get_Field_LAP(By, x, y-4, z);
							stencil_Ay[1] = get_Field_LAP(By, x, y,   z);
							stencil_Ay[2] = get_Field_LAP(By, x, y+4, z);
							
							//for Az parameters
							stencil_Az[0] = get_Field_LAP(Bz, x, y-4, z);
							stencil_Az[1] = get_Field_LAP(Bz, x, y,   z);
							stencil_Az[2] = get_Field_LAP(Bz, x, y+4, z);
							// __threadfence();
						}	
						
					}
				}
			}

		// #pragma unroll 1
		// for(int TEST=0; TEST<1000; TEST++)
		// {
			__syncthreads();
			// __threadfence();
			// /*
			//
			//======================================
			// 			for Y-direction
			//======================================
			{
				/*
				{
					y = threadIdx.z + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y;
					z = threadIdx.y + scanZ*blockIdx.z* blockDim.z                   +job.start.z +ZZZ*4;
					unsigned int offset2 = 1+threadIdx.x+16*threadIdx.z+64*threadIdx.y; // 5
			
					//for rho parameters
					buffer_d[0] = get_Field_LAP(d, x, y, z-4); // 6+1
					buffer_d[1] = Trans_shared[offset2];
					buffer_d[2] = get_Field_LAP(d, x, y, z+4);
					// buffer_d[2] = get_Field_LAP(d, x, y, z+4-flag*8);
					
					//for u parameters
					buffer_u[0] = get_Field_LAP(u, x, y, z-4) ;
					buffer_u[1] = Trans_shared[offset2 +256];
					buffer_u[2] = get_Field_LAP(u, x, y, z+4);
					
					//for v parameters
					buffer_v[0] = get_Field_LAP(v, x, y, z-4) ;
					buffer_v[1] = Trans_shared[offset2 +256*2];
					buffer_v[2] = get_Field_LAP(v, x, y, z+4);
				
					//for w parameters
					buffer_w[0] = get_Field_LAP(w, x, y, z-4) ;
					buffer_w[1] = Trans_shared[offset2 +256*3];
					buffer_w[2] = get_Field_LAP(w, x, y, z+4);
					
					//for T/cc parameters
					buffer_cc[0] = get_Field_LAP(cc, x, y, z-4);
					buffer_cc[1] = Trans_shared[offset2 +256*4];
					buffer_cc[2] = get_Field_LAP(cc, x, y, z+4);
					
					//for Ax parameters
					buffer_Ax[0] = get_Field_LAP(Cx, x, y, z-4) ;
					buffer_Ax[1] = get_Field_LAP(Cx, x, y, z  ) ;
					buffer_Ax[2] = get_Field_LAP(Cx, x, y, z+4);
					
					//for Ay parameters
					buffer_Ay[0] = get_Field_LAP(Cy, x, y, z-4) ;
					buffer_Ay[1] = get_Field_LAP(Cy, x, y, z  ) ;
					buffer_Ay[2] = get_Field_LAP(Cy, x, y, z+4);
					
					//for Az parameters
					buffer_Az[0] = get_Field_LAP(Cz, x, y, z-4) ;
					buffer_Az[1] = get_Field_LAP(Cz, x, y, z  ) ;
					buffer_Az[2] = get_Field_LAP(Cz, x, y, z+4);
				}
				*/
				
				// if( threadIdx.x != (blockDim.x-1) )
				// {
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = x;
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = y;
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = z;
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = stencil_w[1];
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = stencil_cc[1];
				// }
				
				// if( x>=job.start.x && x<=job.end.x && y>=job.start.y && y<=job.end.y && z>=job.start.z && z<=job.end.z)
				// if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-5)
				// /*
				// #pragma unroll 1
				// for(int aaa=0; aaa<1000; aaa++)
				{
					unsigned int offset = threadIdx.y+29*threadIdx.x+464*threadIdx.z; // take y as the continuous direction in LDS. y-z-x: 28*4*16.
					REAL ss[3];
					REAL E1P[3];
					REAL E2P[3];
					REAL E3P[3];
					REAL weno_P[8], weno_M[8];
					
					#pragma unroll 1
					for(int ii=0; ii<3; ii++)
					{
						//-------------		
						// y-dir-fpfm-5
						//-------------
						REAL Ak1, Ak2, Ak3;
						REAL vs, E1, E2, E3;
						// REAL E1P, E2P, E3P;
						REAL E1M, E2M, E3M;
						REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
						
						ss[ii] = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]); // 5
						REAL rcp = 1 / ss[ii];
						Ak1 = stencil_Ax[ii] *rcp;
						Ak2 = stencil_Ay[ii] *rcp;
						Ak3 = stencil_Az[ii] *rcp; // 10*3

						vs = stencil_Ax[ii] * stencil_u[ii] 
						   + stencil_Ay[ii] * stencil_v[ii]
						   + stencil_Az[ii] * stencil_w[ii]; // 5

						E1 = vs;
						E2 = vs - stencil_cc[ii] * ss[ii];
						E3 = vs + stencil_cc[ii] * ss[ii]; // 4

						E1P[ii] = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						E2P[ii] = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						E3P[ii] = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

						E1M = E1 - E1P[ii];
						E2M = E2 - E2P[ii];
						E3M = E3 - E3P[ii]; // 3
						// ----------------------------------------
						tmp0 = stencil_d[ii] *Gamma_d_rcp; // 1
						uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
						uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
						vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
						vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
						wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
						wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3; // 12
						vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
						vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50; // 12
						vv = (Gamma_d - 1.0) * (stencil_u[ii] * stencil_u[ii] 
											  + stencil_v[ii] * stencil_v[ii]
											  + stencil_w[ii] * stencil_w[ii] ); // 7
						W2 = split_C3_d * stencil_cc[ii] * stencil_cc[ii]; // 2
						
						// take z as the continuous direction in LDS. z-y-x: 28*4*16
						REAL fp5, fm5;
						fp5 = tmp0 * (E1P[ii] * vv + E2P[ii] * vvc1 + E3P[ii] * vvc2 + W2 * (E2P[ii] + E3P[ii]));
						fm5 = tmp0 * (E1M     * vv + E2M     * vvc1 + E3M     * vvc2 + W2 * (E2M     + E3M)); // 18
						
						P_shared[4*ii +offset] = fp5;
						M_shared[4*ii +offset] = fm5;
						
						get_SoA_LAP(fpY, x, y+4*ii-4, z, 4) = fp5; // fpfm_5
						get_SoA_LAP(fmY, x, y+4*ii-4, z, 4) = fm5;
					}
					
					// y-dir-fpfm-1
					// for(int ii=0; ii<3; ii++)
					{
						//-------------		
						// y-dir-fpfm-1
						//-------------
						const int ii = 0;
						// REAL ss;
						REAL vs, E1, E2, E3;
						// REAL E1P, E2P, E3P;
						REAL E1M, E2M, E3M;
						REAL tmp0;
						
						// ss[ii] = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]); // 5

						vs = stencil_Ax[ii] * stencil_u[ii] 
						   + stencil_Ay[ii] * stencil_v[ii]
						   + stencil_Az[ii] * stencil_w[ii]; // 5

						E1 = vs;
						E2 = vs - stencil_cc[ii] * ss[ii];
						E3 = vs + stencil_cc[ii] * ss[ii]; // 4

						// E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

						E1M = E1 - E1P[ii];
						E2M = E2 - E2P[ii];
						E3M = E3 - E3P[ii]; // 3
						// ----------------------------------------
						tmp0 = stencil_d[ii] *Gamma_d_rcp; // 1
						
						REAL fp1, fm1;
						
						fp1 = tmp0 * (split_C1_d * E1P[ii] + E2P[ii] + E3P[ii]);
						fm1 = tmp0 * (split_C1_d * E1M     + E2M     + E3M); // 8
						
						P_shared[12 +4*ii +offset] = fp1;
						M_shared[12 +4*ii +offset] = fm1;
						
						get_SoA_LAP(fpY, x, y-4, z, 0) = fp1; // fpfm_1
						get_SoA_LAP(fmY, x, y-4, z, 0) = fm1;
					}
					
					__syncthreads(); // Be Careful this sync ! it might be absolutely necessary.
					// __threadfence();
					
									///////////////////////////
									// buffer_1 prefetching: y_Loop1+: rhs1+
									for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[offset +ii],
															  weno_M[ii] = M_shared[offset +ii];
															  
									// __syncthreads();
									// for(int ii=3; ii<8; ii++) rhs[ii-2] = weno_P[ii];
									// rhs[ii+1] = weno_P[ii], rhs[ii+1] = weno_P[ii], rhs[ii+1] = weno_P[ii], 
									// rhs[ii+1] = weno_P[6], rhs[ii+1] = weno_P[7], ;
					
					// __syncthreads();
					offset -= threadIdx.y;
					REAL Wp_previous = 0.0, Wm_previous = 0.0;
					int point = 1;
					
					#pragma unroll 1
					for(int loop=0; loop<7; loop++)
					{
						// if(loop == 6)
						// {
							// unsigned int offset_tran = threadIdx.z+4*threadIdx.y+16*threadIdx.x;
							// Trans_shared[      offset_tran] = rhs[1];
							// Trans_shared[256  +offset_tran] = rhs[2];
							// Trans_shared[512  +offset_tran] = rhs[3];
							// Trans_shared[768  +offset_tran] = rhs[4];
							// Trans_shared[1024 +offset_tran] = rhs[5];
						// }
						
						#pragma unroll 1
						for(int nn=0; nn<2; nn++)
						{
							// if(loop == 6) break;
							if(loop+nn >= 6) break;
							
							//-------------
							// y-dir-1234
							//-------------
							
							int u3Bits = ( 0b110110110010001000100011010001110000101100 >> 3*(loop*2+nn) ) & 0b111;
								// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("LDS write：%d:  %d\n", (loop*2+nn), u3Bits);
							
							// int u2Bits_2 = ( 0b1001001001001001001001 >> 2*(loop*2+nn) ) & 0b11;
								// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("position of grid：%d:  %d\n", (loop*2+nn), u2Bits_2);
							
							// int u2Bits_1 = ( 0b1111111010100101010000 >> 2*(loop*2+nn) ) & 0b11;
								// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("formula of fpfm：%d:  %d\n", (loop*2+nn), u2Bits_1);
							
							// int ii = u2Bits_2;
							int ii = point % 3;
								// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("position of grid：%d:  %d\n", (loop*2+nn), ii);
							
							REAL Ak1, Ak2, Ak3;
							REAL vs, E1, E2, E3;
							// REAL E1P, E2P, E3P;
							REAL E1M, E2M, E3M;
							REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2;
							
							// ss[ii] = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]); // 5
							REAL rcp = 1 / ss[ii];
							Ak1 = stencil_Ax[ii] *rcp;
							Ak2 = stencil_Ay[ii] *rcp;
							Ak3 = stencil_Az[ii] *rcp; // 10*3

							vs = stencil_Ax[ii] * stencil_u[ii] 
							   + stencil_Ay[ii] * stencil_v[ii]
							   + stencil_Az[ii] * stencil_w[ii]; // 5

							E1 = vs;
							E2 = vs - stencil_cc[ii] * ss[ii];
							E3 = vs + stencil_cc[ii] * ss[ii]; // 4

							// E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
							// E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
							// E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 5*3

							E1M = E1 - E1P[ii];
							E2M = E2 - E2P[ii];
							E3M = E3 - E3P[ii]; // 3
							// ----------------------------------------
							tmp0 = stencil_d[ii] *Gamma_d_rcp;
							uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
							uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
							vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
							vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
							wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
							wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3; // 6*2+1
							
							double N1[4] = {1.0, stencil_u[ii], stencil_v[ii], stencil_w[ii]};
							double N2[4] = {1.0, uc1, vc1, wc1};
							double N3[4] = {1.0, uc2, vc2, wc2};
							
							// int fml = u2Bits_1; // formula
							int fml = point / 3; // formula
							point += 1;
								// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("formula of fpfm：%d:  %d\n", (loop*2+nn), fml);

							REAL fp, fm;
							fp = tmp0 * ( N1[fml]*split_C1_d*E1P[ii]  +  N2[fml]*E2P[ii]  +  N3[fml]*E3P[ii] );
							fm = tmp0 * ( N1[fml]*split_C1_d*E1M      +  N2[fml]*E2M      +  N3[fml]*E3M );// 7*2
							
							P_shared[4*u3Bits +threadIdx.y+offset] = fp;
							M_shared[4*u3Bits +threadIdx.y+offset] = fm;
							
							get_SoA_LAP(fpY, x, y+4*ii-4, z, fml) = fp;
							get_SoA_LAP(fmY, x, y+4*ii-4, z, fml) = fm;
							
										// if(blockIdx.x != (gridDim.x-1) && blockIdx.y != (gridDim.y-1) && blockIdx.z != (gridDim.z-1) 
										// && threadIdx.x != (blockDim.x-1) && threadIdx.y != (blockDim.y-1) && threadIdx.z != (blockDim.z-1))
											// printf("lds offest: %d, threadIdx.y=%d\n", 4*u3Bits +threadIdx.y+offset, threadIdx.y );
						}
						
						// if(loop == 5)
						
						// __syncthreads();
						// __threadfence();
						
						uint32_t rhs_flag[4] = {0b1001000001001001000001000100, 0b1010000001001010100100000100,
												0b1010000001001010101000000100, 0b1010100100001010101010010000};
						int rhs_flag_w = ( rhs_flag[threadIdx.y] >> (loop*4+3) ) & 0b1;
						int rhs_flag_D = ( rhs_flag[threadIdx.y] >> (loop*4+2) ) & 0b1;
						int rhs_flag_U = ( rhs_flag[threadIdx.y] >> (loop*4+1) ) & 0b1;
						int rhs_flag_P = ( rhs_flag[threadIdx.y] >> (loop*4  ) ) & 0b1;
						// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.z == 0)
							// printf("loop: %d, rhs_flag of threadIdx.y=%d： %d,%d,%d,%d\n", loop+1, threadIdx.y, rhs_flag_w, rhs_flag_D, rhs_flag_U, rhs_flag_P );
						
						uint32_t rhs_ID[4] = {0b100000011010000001101, 0b100000011010001000101,
											  0b100000011010001000101, 0b100011000010001101000};
						int rhs_id = ( rhs_ID[threadIdx.y] >> (loop*3) ) & 0b111;
						// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.z == 0)
							// printf("loop: %d, rhs_id of threadIdx.y=%d: %d\n", loop+1, threadIdx.y, rhs_id);
								
								// if(loop == 1)
								// {
									// for(int ii=0; ii<5; ii++) rhs[ii+1] = weno_P[ii];
									// for(int ii=3; ii<8; ii++) rhs[ii-2] = weno_P[ii];
								// }
								// __syncthreads();
								// if(loop == 6 && threadIdx.y == 0)
								// if(loop == 6)
								// {
									// for(int ii=0; ii<5; ii++) rhs[ii+1] = weno_M[ii];
									// for(int ii=3; ii<8; ii++) rhs[ii-2] = weno_P[ii];
								// }
						
								// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.z == 0)
									// printf("loop: %d, threadIdx.y=%d, rhs5 before = ：%lf\n", loop+1, threadIdx.y, rhs[5] );
								
						// if(loop == 5)
						
						
						{
							// y_rhs+
							REAL weno = OCFD_weno7_SYMBO_kernel_P_opt(&weno_P[0]);
							REAL DOWN = __shfl_down_double(weno, 16, hipWarpSize);
							REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
							rhs[rhs_id] += -Ajacobi*( (2*rhs_flag_w-1)*weno +rhs_flag_D*DOWN -rhs_flag_U*UP -rhs_flag_P*Wp_previous )*hy_d_rcp; // 10
							
									// if(loop == 6 && threadIdx.y != 0) rhs[1] = ;
									// if(loop == 6 ) rhs[1] = weno;
									// if(loop == 6 ) rhs[2] = Wp_previous;
									// if(loop == 6 ) rhs[3] = Ajacobi;
									// if(loop == 6 ) rhs[4] = -Ajacobi*( (2*rhs_flag_w-1)*weno +rhs_flag_D*DOWN -rhs_flag_U*UP -rhs_flag_P*Wp_previous )*hy_d_rcp;
									
							Wp_previous = weno;
						}
						
						{
							// y_rhs-
							REAL weno = OCFD_weno7_SYMBO_kernel_M_opt(&weno_M[0]);
							REAL DOWN = __shfl_down_double(weno, 16, hipWarpSize);
							REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
							rhs[rhs_id] += -Ajacobi*( (2*rhs_flag_w-1)*weno +rhs_flag_D*DOWN -rhs_flag_U*UP -rhs_flag_P*Wm_previous )*hy_d_rcp;
							
									// if(loop == 6) rhs[1] = ;
									// if(loop == 6) rhs[1] = weno;
									// if(loop == 6) rhs[2] = Wm_previous;
									// if(loop == 6) rhs[3] = Ajacobi;
									// if(loop == 6) rhs[4] = -Ajacobi*( (2*rhs_flag_w-1)*weno +rhs_flag_D*DOWN -rhs_flag_U*UP -rhs_flag_P*Wm_previous )*hy_d_rcp;
									
							Wm_previous = weno;
						}
						
						int u3Face = ( 0b011010000001000100000 >> 3*loop ) & 0b111; // 0,5,1,2,0,3,4
						int flag_face = ( 0b1101110 >> loop ) & 0b1; // 0,5,1,2,0,3,4
						int flag_thread = threadIdx.y / 3; // 0,0,0,1
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) printf("Face flag：%d:  %d, %d\n", loop, u3Face, flag_face);
						int offset_face = 1+u3Face*64 +threadIdx.x+16*threadIdx.z; // take x as the continuous direction in LDS. x-z: 16*4.
						FaceY_shared[offset_face *flag_face *flag_thread] = Wp_previous + Wm_previous;
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0) printf("Face flag：%d:  %d\n", loop, offset_face *flag_face *flag_thread);
						
						// unsigned int offset_tran = threadIdx.z+4*threadIdx.y+16*threadIdx.x;
						// Trans_shared[256*rhs_id +offset_tran] = rhs[rhs_id];
						
						// __syncthreads();
						// uint8_t offset_lds[8];
						uint64_t lds_load[7][4] = {
							// {0b0011100110001010010000011000100000100000, 0b0100000111001100010100100000110001000001, 0b0100101000001110011000101001000001100010, 0b0101001001010000011100110001010010000011}, // 1
							{0b1001110010100011000001111011100110101100, 0b1010010011100101000110000011110111001101, 
							 0, 0b0101101010010010100000111001100010100100}, // 1
							{0b1101111010110011100000011000100000100000, 0b1010110100100111001010001100000111101110,
							 0b1011010101101001001110010100011000001111, 0b1011110110101011010010011100101000110000}, // 2
							{0b0010011011110101100111000000110001000001, 0b0010100100110111101011001110000001100010,
							 0b0011000101001001101111010110011100000011, 0b0011100110001010010011011110101100111000}, // 3
							{0b0111101110011010110001011010100100101000, 0b1000001111011100110101100010110101001001,
							 0b1000110000011110111001101011000101101010, 0b1001010001100000111101110011010110001011}, // 4
							{0b0011100110001010010000011000100000100000, 0,
							 0 , 0b1001110010100011000001111011100110101100}, // 5
							{0b0100000111001100010100100000110001000001, 0b0100101000001110011000101001000001100010,
							 0b0101001001010000011100110001010010000011, 0b0101101010010010100000111001100010100100}, // 6
							{0, 0, 0, 0} // 7
						};
						
						// #pragma unroll 2
						for(int ii=0; ii<8; ii++)
						{
							if(loop == 6) break;
							// offset_lds[ii] = ( lds_load[loop][threadIdx.y] >> (ii * 5) ) & 0b11111;
							int offset_LDS = ( lds_load[loop][threadIdx.y] >> (ii * 5) ) & 0b11111;
							weno_P[ii] = P_shared[offset_LDS +offset];
							weno_M[ii] = M_shared[offset_LDS +offset];
							
										// if(blockIdx.x == (gridDim.x-1) && blockIdx.y == (gridDim.y-1) && blockIdx.z == (gridDim.z-1) 
										// && threadIdx.x == (blockDim.x-1) && threadIdx.y == (blockDim.y-1) && threadIdx.z == (blockDim.z-1))
												// printf("lds offest: %d, threadIdx.y=%d\n", 1792 +offset_LDS +offset, threadIdx.y );
						}
						
						// if(loop == 5)
						
						// __syncthreads();
						// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.z == 0)
							// printf("loop: %d, loding offest of LDS of threadIdx.y=%d： %d,%d,%d,%d,%d,%d,%d,%d\n", loop+1, threadIdx.y, offset_lds[0], offset_lds[1], offset_lds[2], offset_lds[3], offset_lds[4], offset_lds[5], offset_lds[6], offset_lds[7] );
						
								// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x == 0 && threadIdx.z == 0)
									// printf("loop: %d, threadIdx.y=%d, rhs5 after = ：%lf\n", loop+1, threadIdx.y, rhs[5] );
						
						
					} // loop-7
					
					// __syncthreads();
					
				}
				// */
			}
			// */
			
			
			//======================================
			// 			for Z-direction
			//======================================
			// /*
			{
				y = threadIdx.z + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y;
				z = threadIdx.y + scanZ*blockIdx.z* blockDim.z                   +job.start.z +ZZZ*4;
				
				REAL RHS[6]; for(int ii=0; ii<6; ii++) RHS[ii] = 0.0;
				REAL fpZ_buffer[10], fmZ_buffer[10];
				REAL stencil_d, stencil_u, stencil_v, stencil_w, stencil_cc;
				REAL stencil_Ax, stencil_Ay, stencil_Az;
				unsigned int offset = threadIdx.y+29*threadIdx.x+464*threadIdx.z; // take y as the continuous direction in LDS. y-z-x: 28*4*16.
				
				{
					unsigned int offset_tran = threadIdx.z+4*threadIdx.y+16*threadIdx.x;
					Trans_shared[      offset_tran] = rhs[1];
					Trans_shared[256  +offset_tran] = rhs[2];
					Trans_shared[512  +offset_tran] = rhs[3];
					Trans_shared[768  +offset_tran] = rhs[4];
					Trans_shared[1024 +offset_tran] = rhs[5];
					
					Ajacobi = get_Field_LAP(Ajac, x, y, z);
				}
				
				{
					fpZ_buffer[0] = get_SoA_LAP(fpZ, x, y, z-4, 0); // left
					fpZ_buffer[1] = get_SoA_LAP(fpZ, x, y, z-4, 1);
					fpZ_buffer[2] = get_SoA_LAP(fpZ, x, y, z-4, 2);
					fpZ_buffer[3] = get_SoA_LAP(fpZ, x, y, z-4, 3);
					fpZ_buffer[4] = get_SoA_LAP(fpZ, x, y, z-4, 4);
					fpZ_buffer[5] = get_SoA_LAP(fpZ, x, y, z  , 0); // center
					fpZ_buffer[6] = get_SoA_LAP(fpZ, x, y, z  , 1);
					fpZ_buffer[7] = get_SoA_LAP(fpZ, x, y, z  , 2);
					fpZ_buffer[8] = get_SoA_LAP(fpZ, x, y, z  , 3);
					fpZ_buffer[9] = get_SoA_LAP(fpZ, x, y, z  , 4);
					
					P_shared[  offset] = fpZ_buffer[0]; // left
					P_shared[4+offset] = fpZ_buffer[5]; // center

					fmZ_buffer[0] = get_SoA_LAP(fmZ, x, y, z-4, 0); // left
					fmZ_buffer[1] = get_SoA_LAP(fmZ, x, y, z-4, 1);
					fmZ_buffer[2] = get_SoA_LAP(fmZ, x, y, z-4, 2);
					fmZ_buffer[3] = get_SoA_LAP(fmZ, x, y, z-4, 3);
					fmZ_buffer[4] = get_SoA_LAP(fmZ, x, y, z-4, 4);
					fmZ_buffer[5] = get_SoA_LAP(fmZ, x, y, z  , 0); // center
					fmZ_buffer[6] = get_SoA_LAP(fmZ, x, y, z  , 1);
					fmZ_buffer[7] = get_SoA_LAP(fmZ, x, y, z  , 2);
					fmZ_buffer[8] = get_SoA_LAP(fmZ, x, y, z  , 3);
					fmZ_buffer[9] = get_SoA_LAP(fmZ, x, y, z  , 4);
					
					M_shared[  offset] = fmZ_buffer[0]; // left
					M_shared[4+offset] = fmZ_buffer[5]; // center
					
					stencil_d  = get_Field_LAP(d,  x, y, z+4); // right
					stencil_u  = get_Field_LAP(u,  x, y, z+4);
					stencil_v  = get_Field_LAP(v,  x, y, z+4);
					stencil_w  = get_Field_LAP(w,  x, y, z+4);
					stencil_cc = get_Field_LAP(cc, x, y, z+4);
					
					stencil_Ax = get_Field_LAP(Cx, x, y, z+4);
					stencil_Ay = get_Field_LAP(Cy, x, y, z+4);
					stencil_Az = get_Field_LAP(Cz, x, y, z+4);
				}
				
				// #pragma unroll 1
				// for(int aaa=0; aaa<1000; aaa++)
				{
					REAL ss;
					REAL E1P;
					REAL E2P;
					REAL E3P;
					REAL weno_P[8], weno_M[8];
					
					{
						//-------------		
						// z-dir
						//-------------
						REAL Ak1, Ak2, Ak3;
						REAL vs, E1, E2, E3, E1M, E2M, E3M;
						REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
						
						ss = sqrt(stencil_Ax*stencil_Ax + stencil_Ay*stencil_Ay + stencil_Az*stencil_Az); // 5
						REAL rcp = 1 / ss;
						Ak1 = stencil_Ax *rcp;
						Ak2 = stencil_Ay *rcp;
						Ak3 = stencil_Az *rcp; // 10*3

						vs = stencil_Ax * stencil_u 
						   + stencil_Ay * stencil_v
						   + stencil_Az * stencil_w; // 5

						E1 = vs;
						E2 = vs - stencil_cc * ss;
						E3 = vs + stencil_cc * ss; // 4

						E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

						E1M = E1 - E1P;
						E2M = E2 - E2P;
						E3M = E3 - E3P; // 3
						// ----------------------------------------
						tmp0 = stencil_d *Gamma_d_rcp; // 1
						uc1  = stencil_u - stencil_cc * Ak1; // 2
						uc2  = stencil_u + stencil_cc * Ak1; // 2
						
						REAL fp1,fm1,fp2,fm2;
						
						fp1 = tmp0 * (split_C1_d * E1P + E2P + E3P); // fpfm_1
						fm1 = tmp0 * (split_C1_d * E1M + E2M + E3M);
						fp2 = tmp0 * (split_C1_d * E1P * stencil_u + E2P * uc1 + E3P * uc2); // fpfm_2
						fm2 = tmp0 * (split_C1_d * E1M * stencil_u + E2M * uc1 + E3M * uc2);
						
						P_shared[8  +offset] = fp1; // fpfm_1
						M_shared[8  +offset] = fm1;
						P_shared[12 +offset] = fp2; // fpfm_2
						M_shared[12 +offset] = fm2;
						
						get_SoA_LAP(fpZ, x, y, z+4, 0) = fp1; // fpfm_1
						get_SoA_LAP(fmZ, x, y, z+4, 0) = fm1;
						get_SoA_LAP(fpZ, x, y, z+4, 1) = fp2; // fpfm_2
						get_SoA_LAP(fmZ, x, y, z+4, 1) = fm2;
						
								// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
									// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, tmp0 * (split_C1_d * E1M * stencil_u + E2M * uc1 + E3M * uc2) ); // fpZ_buffer[0]
					}
					
							__syncthreads();
							// __threadfence();
							for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[1+offset +ii],
													  weno_M[ii] = M_shared[1+offset +ii];
					
					{
						//-------------		
						// z-dir
						//-------------
						REAL Ak1, Ak2, Ak3;
						REAL vs, E1, E2, E3, E1M, E2M, E3M;
						REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
						
						// ss = sqrt(stencil_Ax*stencil_Ax + stencil_Ay*stencil_Ay + stencil_Az*stencil_Az); // 5
						REAL rcp = 1 / ss;
						Ak1 = stencil_Ax *rcp;
						Ak2 = stencil_Ay *rcp;
						Ak3 = stencil_Az *rcp; // 10*3

						vs = stencil_Ax * stencil_u 
						   + stencil_Ay * stencil_v
						   + stencil_Az * stencil_w; // 5

						E1 = vs;
						E2 = vs - stencil_cc * ss;
						E3 = vs + stencil_cc * ss; // 4

						// E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

						E1M = E1 - E1P;
						E2M = E2 - E2P;
						E3M = E3 - E3P; // 3
						// ----------------------------------------
						tmp0 = stencil_d *Gamma_d_rcp; // 1
						uc1  = stencil_u - stencil_cc * Ak1;
						uc2  = stencil_u + stencil_cc * Ak1;
						vc1  = stencil_v - stencil_cc * Ak2;
						vc2  = stencil_v + stencil_cc * Ak2;
						wc1  = stencil_w - stencil_cc * Ak3;
						wc2  = stencil_w + stencil_cc * Ak3; // 12
						vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
						vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50; // 12
						vv = (Gamma_d - 1.0) * (stencil_u * stencil_u 
											  + stencil_v * stencil_v
											  + stencil_w * stencil_w ); // 7
						W2 = split_C3_d * stencil_cc * stencil_cc; // 2
						
						// P_shared[8  +offset] = tmp0 * (split_C1_d * E1P + E2P + E3P); // fpfm_1
						// M_shared[8  +offset] = tmp0 * (split_C1_d * E1M + E2M + E3M);
						// P_shared[12 +offset] = tmp0 * (split_C1_d * E1P * stencil_u + E2P * uc1 + E3P * uc2); // fpfm_2
						// M_shared[12 +offset] = tmp0 * (split_C1_d * E1M * stencil_u + E2M * uc1 + E3M * uc2);
						
						REAL fp3,fm3,fp4,fm4,fp5,fm5;
						
						fp3 = tmp0 * (split_C1_d * E1P * stencil_v + E2P * vc1 + E3P * vc2); // fpfm_3
						fm3 = tmp0 * (split_C1_d * E1M * stencil_v + E2M * vc1 + E3M * vc2);
						fp4 = tmp0 * (split_C1_d * E1P * stencil_w + E2P * wc1 + E3P * wc2); // fpfm_4
						fm4 = tmp0 * (split_C1_d * E1M * stencil_w + E2M * wc1 + E3M * wc2);
						fp5 = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P)); // fpfm_5
						fm5 = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
						
						P_shared[16 +offset] = fp3; // fpfm_3
						M_shared[16 +offset] = fm3;
						P_shared[20 +offset] = fp4; // fpfm_4
						M_shared[20 +offset] = fm4;
						P_shared[24 +offset] = fp5; // fpfm_5
						M_shared[24 +offset] = fm5;
						
						get_SoA_LAP(fpZ, x, y, z+4, 2) = fp3; // fpfm_3
						get_SoA_LAP(fmZ, x, y, z+4, 2) = fm3;
						get_SoA_LAP(fpZ, x, y, z+4, 3) = fp4; // fpfm_4
						get_SoA_LAP(fmZ, x, y, z+4, 3) = fm4;
						get_SoA_LAP(fpZ, x, y, z+4, 4) = fp5; // fpfm_5
						get_SoA_LAP(fmZ, x, y, z+4, 4) = fm5;
					}
					
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
								// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, P_shared[12 +offset]);
					
							// __syncthreads();
							P_shared[4+offset] = fpZ_buffer[1]; // fp_2 Left
							P_shared[8+offset] = fpZ_buffer[6]; // fp_2 center
					
					#pragma unroll 1
					for(int loop=1; loop<6; loop++)
					{
						int flag_faceZ = !threadIdx.y; // 1,0,0,0
						REAL weno_plus;
						
						{
							// weno+;
							REAL weno = OCFD_weno7_SYMBO_kernel_P_opt(&weno_P[0]);
							REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
							RHS[loop] += -Ajacobi*( weno - UP*!flag_faceZ )*hz_d_rcp;
							weno_plus = weno;
							
									// if(ZZZ == 1 && loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x==0 && threadIdx.y==1 && threadIdx.z==0)
										// printf("YYY=%d, weno_P: %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E\n", YYY, weno_P[0],weno_P[1],weno_P[2],weno_P[3],weno_P[4],weno_P[5],weno_P[6],weno_P[7]);
									
									// if(ZZZ == 1 && loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
										// printf("YYY=%d, threadIdx.x=%d && threadIdx.y=%d && threadIdx.z=%d,\n weno_P: %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E\n", YYY, threadIdx.x,threadIdx.y,threadIdx.z, 
										// weno_P[0],weno_P[1],weno_P[2],weno_P[3],weno_P[4],weno_P[5],weno_P[6],weno_P[7]);
										
									// if(ZZZ == 1 && loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
										// printf("YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", YYY, threadIdx.x,threadIdx.y,threadIdx.z, weno);
									
									// if(loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
										// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, fpZ_buffer[0]); // fpZ_buffer[0]
						}
						
								if(loop<5)
								{
									M_shared[  loop*4+offset] = fmZ_buffer[loop], 
									M_shared[4+loop*4+offset] = fmZ_buffer[loop+5]; // WRITE: fm_2345*(左中)
									for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[1+loop*4+offset +ii]; // READ:  fp_2345*(左中右)
								}
					
						int offset_Previous = -63+loop*64 +threadIdx.x+16*threadIdx.z; // take x as the continuous direction in LDS. x-z: 16*4.
						REAL weno_Previous = FaceZ_shared[offset_Previous];
						
						{
							// weno-;
							REAL weno = OCFD_weno7_SYMBO_kernel_M_opt(&weno_M[0]);
							REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
							RHS[loop] += -Ajacobi*( weno - UP*!flag_faceZ )*hz_d_rcp;
							int flag_thread = threadIdx.y / 3; // 0,0,0,1
							FaceZ_shared[offset_Previous *flag_thread] = weno_plus + weno;
							
									// if(ZZZ == 1 && loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
										// printf("YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", YYY, threadIdx.x,threadIdx.y,threadIdx.z, weno);
									// if(loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
										// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, weno);
									// if(loop == 1)
										// printf("ZZZ=%d, YYY=%d; blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, blockIdx.x,blockIdx.y,blockIdx.z, threadIdx.x,threadIdx.y,threadIdx.z, weno);
									
									// if(loop == 5 && blockIdx.x==1 && blockIdx.y==1 && blockIdx.z==1)
										// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, weno);
						}
						RHS[loop] += Ajacobi *weno_Previous *hz_d_rcp *flag_faceZ;
						
								int flag_loop = loop/4; // 0,0,0,1,1
								if(loop<5)
								{
									P_shared[ (4+loop*4+offset)* !flag_loop] = fpZ_buffer[loop+1], 
									P_shared[ (8+loop*4+offset)* !flag_loop] = fpZ_buffer[loop+6]; // WRITE: fp_345**(左中)
									for(int ii=0; ii<8; ii++) weno_M[ii] = M_shared[1+loop*4+offset +ii]; // READ:  fm_2345*(左中右)
								}
							
							
									// if(loop == 4 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
										// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, RHS[loop] );
					}
				}
				
				unsigned int offset_tran = threadIdx.y+4*threadIdx.z+16*threadIdx.x;
				if( threadIdx.x != (blockDim.x-1) )
				{
					get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = RHS[1] +Trans_shared[      offset_tran];
					get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = RHS[2] +Trans_shared[256  +offset_tran];
					get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = RHS[3] +Trans_shared[512  +offset_tran];
					get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = RHS[4] +Trans_shared[768  +offset_tran];
					get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = RHS[5] +Trans_shared[1024 +offset_tran];
					
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = RHS[1] +rhs[1];
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = RHS[2] +rhs[2];
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = RHS[3] +rhs[3];
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = RHS[4] +rhs[4];
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = RHS[5] +rhs[5];
				}
			}
			
		} // Second line first tile.
		
		// __syncthreads();
		
		// Second line YYY loop.
		#pragma unroll 1
		for(int YYY=1; YYY<scanY; YYY++)
		{
			__syncthreads();
			unsigned int x = threadIdx.x +       blockIdx.x*(blockDim.x-1) + blockDim.x-1 +job.start.x; // with x overlap, inner point only.
			unsigned int y = threadIdx.y + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y +YYY*4; // inner point only.
			unsigned int z = threadIdx.z + scanZ*blockIdx.z* blockDim.z                   +job.start.z +ZZZ*4;
			
			REAL fpY_buffer[10], fmY_buffer[10];
			REAL fpZ_buffer[10], fmZ_buffer[10];
			
			REAL yStencil_d, yStencil_u, yStencil_v, yStencil_w, yStencil_cc;
			REAL yStencil_Ax, yStencil_Ay, yStencil_Az;
			
			REAL zStencil_d, zStencil_u, zStencil_v, zStencil_w, zStencil_cc;
			REAL zStencil_Ax, zStencil_Ay, zStencil_Az;
			
			REAL Ajacobi;
			REAL rhs[6]; for(int ii=0; ii<6; ii++) rhs[ii] = 0.0; // 0 is dummy, 1~5 is real.
			
			//======================================
			// 			for X-direction
			//======================================
			// /*
			{
				REAL xStencil_d[2]; 
				REAL xStencil_u[2]; 
				REAL xStencil_v[2]; 
				REAL xStencil_w[2]; 
				REAL xStencil_cc[2];
				
				REAL xStencil_Ax[2];
				REAL xStencil_Ay[2];
				REAL xStencil_Az[2];
				
				{
					int flag1 = (blockDim.x-4 +threadIdx.x) / blockDim.x; // threadIdx.x < 4,  flag = 0; otherwise flag = 1.
					int flag2 = (blockDim.x+12-threadIdx.x) / blockDim.x; // threadIdx.x > 12, flag = 0; otherwise flag = 1.
					
					// unsigned int offset = 1 +threadIdx.x+16*threadIdx.y +64*threadIdx.z;
					
					//
					//for rho parameters
					xStencil_d[0] = get_Field_LAP(d, x-4, y, z) ;
					xStencil_d[1] = get_Field_LAP(d, x+3, y, z) ;
					
					//
					//for u parameters
					xStencil_u[0] = get_Field_LAP(u, x-4, y, z) ;
					xStencil_u[1] = get_Field_LAP(u, x+3, y, z) ;
					
					//
					//for v parameters
					xStencil_v[0] = get_Field_LAP(v, x-4, y, z) ;
					xStencil_v[1] = get_Field_LAP(v, x+3, y, z) ;
					
					//
					//for w parameters
					xStencil_w[0] = get_Field_LAP(w, x-4, y, z) ;
					xStencil_w[1] = get_Field_LAP(w, x+3, y, z) ;
					
					//
					//for T/cc parameters
					xStencil_cc[0] = get_Field_LAP(cc, x-4, y, z);
					xStencil_cc[1] = get_Field_LAP(cc, x+3, y, z);
					
					//
					//for Ax parameters
					xStencil_Ax[0] = get_Field_LAP(Ax, x-4, y, z);
					xStencil_Ax[1] = get_Field_LAP(Ax, x+3, y, z);
					
					//
					//for Ay parameters
					xStencil_Ay[0] = get_Field_LAP(Ay, x-4, y, z);
					xStencil_Ay[1] = get_Field_LAP(Ay, x+3, y, z);
					
					//
					//for Az parameters
					xStencil_Az[0] = get_Field_LAP(Az, x-4, y, z);
					xStencil_Az[1] = get_Field_LAP(Az, x+3, y, z);
					
					//
					//for jacobian parameters
					Ajacobi = get_Field_LAP(Ajac, x, y, z);
				}
				
				// #pragma unroll 1
				// for(int aaa=0; aaa<1000; aaa++)
				{
					// int offset = threadIdx.x+24*threadIdx.y+96*threadIdx.z; // take x as the continuous direction in LDS. x-y-z: 24*4*4 = 384.
					unsigned int offset = threadIdx.x+23*threadIdx.y+92*threadIdx.z; // take x as the continuous direction in LDS. x-y-z: 23*4*4 = 368.
					// REAL ss[3];
					// REAL E1P[3];
					// REAL E2P[3];
					// REAL E3P[3];
					
					#pragma unroll 1
					for(int ii=0; ii<2; ii++)
					{
						//-------------		
						// x-dir
						//-------------
						REAL ss, Ak1, Ak2, Ak3;
						REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
						REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
						
						ss = sqrt(xStencil_Ax[ii]*xStencil_Ax[ii] + xStencil_Ay[ii]*xStencil_Ay[ii] + xStencil_Az[ii]*xStencil_Az[ii]); // 5
						REAL rcp = 1 / ss;
						Ak1 = xStencil_Ax[ii] *rcp;
						Ak2 = xStencil_Ay[ii] *rcp;
						Ak3 = xStencil_Az[ii] *rcp; // 10*3

						vs = xStencil_Ax[ii] * xStencil_u[ii] 
						   + xStencil_Ay[ii] * xStencil_v[ii]
						   + xStencil_Az[ii] * xStencil_w[ii]; // 5

						E1 = vs;
						E2 = vs - xStencil_cc[ii] * ss;
						E3 = vs + xStencil_cc[ii] * ss; // 4

						E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

						E1M = E1 - E1P;
						E2M = E2 - E2P;
						E3M = E3 - E3P; // 3
						// ----------------------------------------
						tmp0 = xStencil_d[ii] *Gamma_d_rcp; // 1
						uc1  = xStencil_u[ii] - xStencil_cc[ii] * Ak1;
						uc2  = xStencil_u[ii] + xStencil_cc[ii] * Ak1;
						vc1  = xStencil_v[ii] - xStencil_cc[ii] * Ak2;
						vc2  = xStencil_v[ii] + xStencil_cc[ii] * Ak2;
						wc1  = xStencil_w[ii] - xStencil_cc[ii] * Ak3;
						wc2  = xStencil_w[ii] + xStencil_cc[ii] * Ak3; // 12
						vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
						vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50; // 12
						vv = (Gamma_d - 1.0) * (xStencil_u[ii] * xStencil_u[ii] 
											  + xStencil_v[ii] * xStencil_v[ii]
											  + xStencil_w[ii] * xStencil_w[ii] ); // 7
						W2 = split_C3_d * xStencil_cc[ii] * xStencil_cc[ii]; // 2
						
						P_shared[     7*ii +offset] = tmp0 * (split_C1_d * E1P + E2P + E3P); // 1
						M_shared[     7*ii +offset] = tmp0 * (split_C1_d * E1M + E2M + E3M);
						P_shared[ 368+7*ii +offset] = tmp0 * (split_C1_d * E1P * xStencil_u[ii] + E2P * uc1 + E3P * uc2); // 2
						M_shared[ 368+7*ii +offset] = tmp0 * (split_C1_d * E1M * xStencil_u[ii] + E2M * uc1 + E3M * uc2);
						P_shared[ 736+7*ii +offset] = tmp0 * (split_C1_d * E1P * xStencil_v[ii] + E2P * vc1 + E3P * vc2); // 3
						M_shared[ 736+7*ii +offset] = tmp0 * (split_C1_d * E1M * xStencil_v[ii] + E2M * vc1 + E3M * vc2);
						P_shared[1104+7*ii +offset] = tmp0 * (split_C1_d * E1P * xStencil_w[ii] + E2P * wc1 + E3P * wc2); // 4
						M_shared[1104+7*ii +offset] = tmp0 * (split_C1_d * E1M * xStencil_w[ii] + E2M * wc1 + E3M * wc2);
						P_shared[1472+7*ii +offset] = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P)); // 5
						M_shared[1472+7*ii +offset] = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
					}
					
							// /*
								{
									fpY_buffer[0] = get_SoA_LAP(fpY, x, y-4, z, 0); // left
									fpY_buffer[1] = get_SoA_LAP(fpY, x, y-4, z, 1);
									// fpY_buffer[2] = get_SoA_LAP(fpY, x, y-4, z, 2);
									// fpY_buffer[3] = get_SoA_LAP(fpY, x, y-4, z, 3);
									// fpY_buffer[4] = get_SoA_LAP(fpY, x, y-4, z, 4);
									fpY_buffer[5] = get_SoA_LAP(fpY, x, y  , z, 0); // center
									fpY_buffer[6] = get_SoA_LAP(fpY, x, y  , z, 1);
									// fpY_buffer[7] = get_SoA_LAP(fpY, x, y  , z, 2);
									// fpY_buffer[8] = get_SoA_LAP(fpY, x, y  , z, 3);
									// fpY_buffer[9] = get_SoA_LAP(fpY, x, y  , z, 4);

									fmY_buffer[0] = get_SoA_LAP(fmY, x, y-4, z, 0); // left
									fmY_buffer[1] = get_SoA_LAP(fmY, x, y-4, z, 1);
									// fmY_buffer[2] = get_SoA_LAP(fmY, x, y-4, z, 2);
									// fmY_buffer[3] = get_SoA_LAP(fmY, x, y-4, z, 3);
									// fmY_buffer[4] = get_SoA_LAP(fmY, x, y-4, z, 4);
									fmY_buffer[5] = get_SoA_LAP(fmY, x, y  , z, 0); // center
									fmY_buffer[6] = get_SoA_LAP(fmY, x, y  , z, 1);
									// fmY_buffer[7] = get_SoA_LAP(fmY, x, y  , z, 2);
									// fmY_buffer[8] = get_SoA_LAP(fmY, x, y  , z, 3);
									// fmY_buffer[9] = get_SoA_LAP(fmY, x, y  , z, 4);
									
									yStencil_d  = get_Field_LAP(d,  x, y+4, z); // right
									yStencil_u  = get_Field_LAP(u,  x, y+4, z);
									yStencil_v  = get_Field_LAP(v,  x, y+4, z);
									yStencil_w  = get_Field_LAP(w,  x, y+4, z);
									yStencil_cc = get_Field_LAP(cc, x, y+4, z);
									
									yStencil_Ax = get_Field_LAP(Bx, x, y+4, z);
									yStencil_Ay = get_Field_LAP(By, x, y+4, z);
									yStencil_Az = get_Field_LAP(Bz, x, y+4, z);
								}
							// */
					
					// __threadfence();
					// __syncthreads();
						
					REAL weno_P[8], weno_M[8];
					// #pragma unroll 1
					for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[offset +ii],
											  weno_M[ii] = M_shared[offset +ii];
					
					#pragma unroll 1
					for(int Loop=1; Loop<6; Loop++)
					{
						{
							// x_rhs1+
							REAL tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&weno_P[0]);
							REAL tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
							rhs[Loop] += -Ajacobi*(tmp_r - tmp_l)*hx_d_rcp; // put_du_p_kernel
						}
						for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[368*Loop +offset +ii];
						
						{
							// x_rhs1-
							REAL tmp_l = OCFD_weno7_SYMBO_kernel_M_opt (&weno_M[0]);
							REAL tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
							rhs[Loop] += -Ajacobi*(tmp_r - tmp_l)*hx_d_rcp; // put_du_m_kernel
						}
						for(int ii=0; ii<8; ii++) weno_M[ii] = M_shared[368*Loop +offset +ii];
						
						// if(Loop == 3)
					}
				}
			}

		// #pragma unroll 1
		// for(int TEST=0; TEST<1000; TEST++)
		// {
			// /*
			//
			//======================================
			// 			for Y-direction
			//======================================
			{
				// REAL fpY_buffer[10], fmY_buffer[10];
				// REAL yStencil_d, yStencil_u, yStencil_v, yStencil_w, yStencil_cc;
				// REAL yStencil_Ax, yStencil_Ay, yStencil_Az;
				unsigned int offset = threadIdx.y+29*threadIdx.x+464*threadIdx.z; // take y as the continuous direction in LDS. y-z-x: 28*4*16.
				
				__syncthreads();
				// __threadfence();
				{
					// fpY_buffer[0] = get_SoA_LAP(fpY, x, y-4, z, 0); // left
					// fpY_buffer[1] = get_SoA_LAP(fpY, x, y-4, z, 1);
					fpY_buffer[2] = get_SoA_LAP(fpY, x, y-4, z, 2);
					fpY_buffer[3] = get_SoA_LAP(fpY, x, y-4, z, 3);
					fpY_buffer[4] = get_SoA_LAP(fpY, x, y-4, z, 4);
					// fpY_buffer[5] = get_SoA_LAP(fpY, x, y  , z, 0); // center
					// fpY_buffer[6] = get_SoA_LAP(fpY, x, y  , z, 1);
					fpY_buffer[7] = get_SoA_LAP(fpY, x, y  , z, 2);
					fpY_buffer[8] = get_SoA_LAP(fpY, x, y  , z, 3);
					fpY_buffer[9] = get_SoA_LAP(fpY, x, y  , z, 4);
					
					P_shared[  offset] = fpY_buffer[0]; // left
					P_shared[4+offset] = fpY_buffer[5]; // center

					// fmY_buffer[0] = get_SoA_LAP(fmY, x, y-4, z, 0); // left
					// fmY_buffer[1] = get_SoA_LAP(fmY, x, y-4, z, 1);
					fmY_buffer[2] = get_SoA_LAP(fmY, x, y-4, z, 2);
					fmY_buffer[3] = get_SoA_LAP(fmY, x, y-4, z, 3);
					fmY_buffer[4] = get_SoA_LAP(fmY, x, y-4, z, 4);
					// fmY_buffer[5] = get_SoA_LAP(fmY, x, y  , z, 0); // center
					// fmY_buffer[6] = get_SoA_LAP(fmY, x, y  , z, 1);
					fmY_buffer[7] = get_SoA_LAP(fmY, x, y  , z, 2);
					fmY_buffer[8] = get_SoA_LAP(fmY, x, y  , z, 3);
					fmY_buffer[9] = get_SoA_LAP(fmY, x, y  , z, 4);
					
					M_shared[  offset] = fmY_buffer[0]; // left
					M_shared[4+offset] = fmY_buffer[5]; // center
					
					// yStencil_d  = get_Field_LAP(d,  x, y+4, z); // right
					// yStencil_u  = get_Field_LAP(u,  x, y+4, z);
					// yStencil_v  = get_Field_LAP(v,  x, y+4, z);
					// yStencil_w  = get_Field_LAP(w,  x, y+4, z);
					// yStencil_cc = get_Field_LAP(cc, x, y+4, z);
					
					// yStencil_Ax = get_Field_LAP(Bx, x, y+4, z);
					// yStencil_Ay = get_Field_LAP(By, x, y+4, z);
					// yStencil_Az = get_Field_LAP(Bz, x, y+4, z);
				}
				
				/*
				{
					unsigned int Y = threadIdx.z + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y +YYY*4;
					unsigned int Z = threadIdx.y + scanZ*blockIdx.z* blockDim.z                   +job.start.z +ZZZ*4;
					
					// fpZ_buffer[0] = get_SoA_LAP(fpZ, x, y, z-4, 0); // left
					// fpZ_buffer[1] = get_SoA_LAP(fpZ, x, y, z-4, 1);
					// fpZ_buffer[2] = get_SoA_LAP(fpZ, x, y, z-4, 2);
					// fpZ_buffer[3] = get_SoA_LAP(fpZ, x, y, z-4, 3);
					// fpZ_buffer[4] = get_SoA_LAP(fpZ, x, y, z-4, 4);
					// fpZ_buffer[5] = get_SoA_LAP(fpZ, x, y, z  , 0); // center
					// fpZ_buffer[6] = get_SoA_LAP(fpZ, x, y, z  , 1);
					// fpZ_buffer[7] = get_SoA_LAP(fpZ, x, y, z  , 2);
					// fpZ_buffer[8] = get_SoA_LAP(fpZ, x, y, z  , 3);
					// fpZ_buffer[9] = get_SoA_LAP(fpZ, x, y, z  , 4);

					// fmZ_buffer[0] = get_SoA_LAP(fmZ, x, y, z-4, 0); // left
					// fmZ_buffer[1] = get_SoA_LAP(fmZ, x, y, z-4, 1);
					// fmZ_buffer[2] = get_SoA_LAP(fmZ, x, y, z-4, 2);
					// fmZ_buffer[3] = get_SoA_LAP(fmZ, x, y, z-4, 3);
					// fmZ_buffer[4] = get_SoA_LAP(fmZ, x, y, z-4, 4);
					// fmZ_buffer[5] = get_SoA_LAP(fmZ, x, y, z  , 0); // center
					// fmZ_buffer[6] = get_SoA_LAP(fmZ, x, y, z  , 1);
					// fmZ_buffer[7] = get_SoA_LAP(fmZ, x, y, z  , 2);
					// fmZ_buffer[8] = get_SoA_LAP(fmZ, x, y, z  , 3);
					// fmZ_buffer[9] = get_SoA_LAP(fmZ, x, y, z  , 4);
					
					zStencil_d  = get_Field_LAP(d,  x, Y, Z+4); // right
					zStencil_u  = get_Field_LAP(u,  x, Y, Z+4);
					zStencil_v  = get_Field_LAP(v,  x, Y, Z+4);
					zStencil_w  = get_Field_LAP(w,  x, Y, Z+4);
					zStencil_cc = get_Field_LAP(cc, x, Y, Z+4);
					
					zStencil_Ax = get_Field_LAP(Cx, x, Y, Z+4);
					zStencil_Ay = get_Field_LAP(Cy, x, Y, Z+4);
					zStencil_Az = get_Field_LAP(Cz, x, Y, Z+4);
				}
				*/
				
				// if( threadIdx.x != (blockDim.x-1) )
				// {
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = x;
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = y;
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = z;
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = yStencil_w[1];
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = yStencil_cc[1];
				// }
				
				// /*
				// for(int aaa=0; aaa<1000; aaa++)
				{
					// y = threadIdx.y + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y +YYY*4; // inner point only.
					// z = threadIdx.z + scanZ*blockIdx.z* blockDim.z                   +job.start.z +ZZZ*4;
					
					REAL ss;
					REAL E1P;
					REAL E2P;
					REAL E3P;
					REAL weno_P[8], weno_M[8];
					
					// __syncthreads();
					{
						//-------------		
						// y-dir
						//-------------
						REAL Ak1, Ak2, Ak3;
						REAL vs, E1, E2, E3, E1M, E2M, E3M;
						REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
						
						ss = sqrt(yStencil_Ax*yStencil_Ax + yStencil_Ay*yStencil_Ay + yStencil_Az*yStencil_Az); // 5
						REAL rcp = 1 / ss;
						Ak1 = yStencil_Ax *rcp;
						Ak2 = yStencil_Ay *rcp;
						Ak3 = yStencil_Az *rcp; // 10*3

						vs = yStencil_Ax * yStencil_u 
						   + yStencil_Ay * yStencil_v
						   + yStencil_Az * yStencil_w; // 5

						E1 = vs;
						E2 = vs - yStencil_cc * ss;
						E3 = vs + yStencil_cc * ss; // 4

						E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

						E1M = E1 - E1P;
						E2M = E2 - E2P;
						E3M = E3 - E3P; // 3
						// ----------------------------------------
						tmp0 = yStencil_d *Gamma_d_rcp; // 1
						uc1  = yStencil_u - yStencil_cc * Ak1; // 2
						uc2  = yStencil_u + yStencil_cc * Ak1; // 2
						
						REAL fp1,fm1,fp2,fm2;
						
						fp1 = tmp0 * (split_C1_d * E1P + E2P + E3P); // fpfm_1
						fm1 = tmp0 * (split_C1_d * E1M + E2M + E3M);
						fp2 = tmp0 * (split_C1_d * E1P * yStencil_u + E2P * uc1 + E3P * uc2); // fpfm_2
						fm2 = tmp0 * (split_C1_d * E1M * yStencil_u + E2M * uc1 + E3M * uc2);
						
						P_shared[8  +offset] = fp1; // fpfm_1
						M_shared[8  +offset] = fm1;
						P_shared[12 +offset] = fp2; // fpfm_2
						M_shared[12 +offset] = fm2;
						
						get_SoA_LAP(fpY, x, y+4, z, 0) = fp1; // fpfm_1
						get_SoA_LAP(fmY, x, y+4, z, 0) = fm1;
						get_SoA_LAP(fpY, x, y+4, z, 1) = fp2; // fpfm_2
						get_SoA_LAP(fmY, x, y+4, z, 1) = fm2;
						
						
								// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
									// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, tmp0 * (split_C1_d * E1M * yStencil_u + E2M * uc1 + E3M * uc2) ); // fpY_buffer[0]
					}
					
							__syncthreads();
							// __threadfence();
							for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[1+offset +ii],
													  weno_M[ii] = M_shared[1+offset +ii];
					
					{
						//-------------		
						// y-dir
						//-------------
						REAL Ak1, Ak2, Ak3;
						REAL vs, E1, E2, E3, E1M, E2M, E3M;
						REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
						
						// ss = sqrt(yStencil_Ax*yStencil_Ax + yStencil_Ay*yStencil_Ay + yStencil_Az*yStencil_Az); // 5
						REAL rcp = 1 / ss;
						Ak1 = yStencil_Ax *rcp;
						Ak2 = yStencil_Ay *rcp;
						Ak3 = yStencil_Az *rcp; // 10*3

						vs = yStencil_Ax * yStencil_u 
						   + yStencil_Ay * yStencil_v
						   + yStencil_Az * yStencil_w; // 5

						E1 = vs;
						E2 = vs - yStencil_cc * ss;
						E3 = vs + yStencil_cc * ss; // 4

						// E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

						E1M = E1 - E1P;
						E2M = E2 - E2P;
						E3M = E3 - E3P; // 3
						// ----------------------------------------
						tmp0 = yStencil_d *Gamma_d_rcp; // 1
						uc1  = yStencil_u - yStencil_cc * Ak1;
						uc2  = yStencil_u + yStencil_cc * Ak1;
						vc1  = yStencil_v - yStencil_cc * Ak2;
						vc2  = yStencil_v + yStencil_cc * Ak2;
						wc1  = yStencil_w - yStencil_cc * Ak3;
						wc2  = yStencil_w + yStencil_cc * Ak3; // 12
						vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
						vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50; // 12
						vv = (Gamma_d - 1.0) * (yStencil_u * yStencil_u 
											  + yStencil_v * yStencil_v
											  + yStencil_w * yStencil_w ); // 7
						W2 = split_C3_d * yStencil_cc * yStencil_cc; // 2
						
						// P_shared[8  +offset] = tmp0 * (split_C1_d * E1P + E2P + E3P); // fpfm_1
						// M_shared[8  +offset] = tmp0 * (split_C1_d * E1M + E2M + E3M);
						// P_shared[12 +offset] = tmp0 * (split_C1_d * E1P * yStencil_u + E2P * uc1 + E3P * uc2); // fpfm_2
						// M_shared[12 +offset] = tmp0 * (split_C1_d * E1M * yStencil_u + E2M * uc1 + E3M * uc2);
						
						REAL fp3,fm3,fp4,fm4,fp5,fm5;
						
						fp3 = tmp0 * (split_C1_d * E1P * yStencil_v + E2P * vc1 + E3P * vc2); // fpfm_3
						fm3 = tmp0 * (split_C1_d * E1M * yStencil_v + E2M * vc1 + E3M * vc2);
						fp4 = tmp0 * (split_C1_d * E1P * yStencil_w + E2P * wc1 + E3P * wc2); // fpfm_4
						fm4 = tmp0 * (split_C1_d * E1M * yStencil_w + E2M * wc1 + E3M * wc2);
						fp5 = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P)); // fpfm_5
						fm5 = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
						
						P_shared[16 +offset] = fp3; // fpfm_3
						M_shared[16 +offset] = fm3;
						P_shared[20 +offset] = fp4; // fpfm_4
						M_shared[20 +offset] = fm4;
						P_shared[24 +offset] = fp5; // fpfm_5
						M_shared[24 +offset] = fm5;
						
						get_SoA_LAP(fpY, x, y+4, z, 2) = fp3; // fpfm_3
						get_SoA_LAP(fmY, x, y+4, z, 2) = fm3;
						get_SoA_LAP(fpY, x, y+4, z, 3) = fp4; // fpfm_4
						get_SoA_LAP(fmY, x, y+4, z, 3) = fm4;
						get_SoA_LAP(fpY, x, y+4, z, 4) = fp5; // fpfm_5
						get_SoA_LAP(fmY, x, y+4, z, 4) = fm5;
					}
					
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
								// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, P_shared[12 +offset]);
					
							// __syncthreads();
							P_shared[4+offset] = fpY_buffer[1]; // fp_2 Left
							P_shared[8+offset] = fpY_buffer[6]; // fp_2 center
					
					#pragma unroll 1
					for(int loop=1; loop<6; loop++)
					{
						int flag_faceY = !threadIdx.y; // 1,0,0,0
						REAL weno_plus;
						
						{
							// weno+;
							REAL weno = OCFD_weno7_SYMBO_kernel_P_opt(&weno_P[0]);
							REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
							rhs[loop] += -Ajacobi*( weno - UP*!flag_faceY )*hy_d_rcp;
							weno_plus = weno;
						}
						
								if(loop<5)
								{
									M_shared[  loop*4+offset] = fmY_buffer[loop], 
									M_shared[4+loop*4+offset] = fmY_buffer[loop+5]; // WRITE: fm_2345*(左中)
									for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[1+loop*4+offset +ii]; // READ:  fp_2345*(左中右)
								}
					
						int offset_Previous = -63+loop*64 +threadIdx.x+16*threadIdx.z; // take x as the continuous direction in LDS. x-z: 16*4.
						REAL weno_Previous = FaceY_shared[offset_Previous];
						
						{
							// weno-;
							REAL weno = OCFD_weno7_SYMBO_kernel_M_opt(&weno_M[0]);
							REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
							rhs[loop] += -Ajacobi*( weno - UP*!flag_faceY )*hy_d_rcp;
							int flag_thread = threadIdx.y / 3; // 0,0,0,1
							FaceY_shared[offset_Previous *flag_thread] = weno_plus + weno;
						}
						rhs[loop] += Ajacobi *weno_Previous *hy_d_rcp *flag_faceY;
						
								int flag_loop = loop/4; // 0,0,0,1,1
								if(loop<5)
								{
									P_shared[ (4+loop*4+offset)* !flag_loop] = fpY_buffer[loop+1], 
									P_shared[ (8+loop*4+offset)* !flag_loop] = fpY_buffer[loop+6]; // WRITE: fp_345**(左中)
									for(int ii=0; ii<8; ii++) weno_M[ii] = M_shared[1+loop*4+offset +ii]; // READ:  fm_2345*(左中右)
								}
					}
				}
				// */
			}
			// */
			
			
			//======================================
			// 			for Z-direction
			//======================================
			// /*
			{
				// __syncthreads();
				y = threadIdx.z + scanY*blockIdx.y* blockDim.y    + blockDim.y   +job.start.y +YYY*4;
				z = threadIdx.y + scanZ*blockIdx.z* blockDim.z                   +job.start.z +ZZZ*4;
				
				// REAL fpZ_buffer[10], fmZ_buffer[10];
				// REAL zStencil_d, zStencil_u, zStencil_v, zStencil_w, zStencil_cc;
				// REAL zStencil_Ax, zStencil_Ay, zStencil_Az;
				REAL RHS[6]; for(int ii=0; ii<6; ii++) RHS[ii] = 0.0;
				
				{
					unsigned int offset_tran = threadIdx.z+4*threadIdx.y+16*threadIdx.x;
					Trans_shared[      offset_tran] = rhs[1];
					Trans_shared[256  +offset_tran] = rhs[2];
					Trans_shared[512  +offset_tran] = rhs[3];
					Trans_shared[768  +offset_tran] = rhs[4];
					Trans_shared[1024 +offset_tran] = rhs[5];
					
					Ajacobi = get_Field_LAP(Ajac, x, y, z);
				}
				
				unsigned int offset = threadIdx.y+29*threadIdx.x+464*threadIdx.z; // take y as the continuous direction in LDS. y-z-x: 28*4*16.
				
				{
					fpZ_buffer[0] = get_SoA_LAP(fpZ, x, y, z-4, 0); // left
					fpZ_buffer[1] = get_SoA_LAP(fpZ, x, y, z-4, 1);
					fpZ_buffer[2] = get_SoA_LAP(fpZ, x, y, z-4, 2);
					fpZ_buffer[3] = get_SoA_LAP(fpZ, x, y, z-4, 3);
					fpZ_buffer[4] = get_SoA_LAP(fpZ, x, y, z-4, 4);
					fpZ_buffer[5] = get_SoA_LAP(fpZ, x, y, z  , 0); // center
					fpZ_buffer[6] = get_SoA_LAP(fpZ, x, y, z  , 1);
					fpZ_buffer[7] = get_SoA_LAP(fpZ, x, y, z  , 2);
					fpZ_buffer[8] = get_SoA_LAP(fpZ, x, y, z  , 3);
					fpZ_buffer[9] = get_SoA_LAP(fpZ, x, y, z  , 4);
					
					P_shared[  offset] = fpZ_buffer[0]; // left
					P_shared[4+offset] = fpZ_buffer[5]; // center

					fmZ_buffer[0] = get_SoA_LAP(fmZ, x, y, z-4, 0); // left
					fmZ_buffer[1] = get_SoA_LAP(fmZ, x, y, z-4, 1);
					fmZ_buffer[2] = get_SoA_LAP(fmZ, x, y, z-4, 2);
					fmZ_buffer[3] = get_SoA_LAP(fmZ, x, y, z-4, 3);
					fmZ_buffer[4] = get_SoA_LAP(fmZ, x, y, z-4, 4);
					fmZ_buffer[5] = get_SoA_LAP(fmZ, x, y, z  , 0); // center
					fmZ_buffer[6] = get_SoA_LAP(fmZ, x, y, z  , 1);
					fmZ_buffer[7] = get_SoA_LAP(fmZ, x, y, z  , 2);
					fmZ_buffer[8] = get_SoA_LAP(fmZ, x, y, z  , 3);
					fmZ_buffer[9] = get_SoA_LAP(fmZ, x, y, z  , 4);
					
					M_shared[  offset] = fmZ_buffer[0]; // left
					M_shared[4+offset] = fmZ_buffer[5]; // center
					
					zStencil_d  = get_Field_LAP(d,  x, y, z+4); // right
					zStencil_u  = get_Field_LAP(u,  x, y, z+4);
					zStencil_v  = get_Field_LAP(v,  x, y, z+4);
					zStencil_w  = get_Field_LAP(w,  x, y, z+4);
					zStencil_cc = get_Field_LAP(cc, x, y, z+4);
					
					zStencil_Ax = get_Field_LAP(Cx, x, y, z+4);
					zStencil_Ay = get_Field_LAP(Cy, x, y, z+4);
					zStencil_Az = get_Field_LAP(Cz, x, y, z+4);
				}
				
				// #pragma unroll 1
				// for(int aaa=0; aaa<1000; aaa++)
				{
					REAL ss;
					REAL E1P;
					REAL E2P;
					REAL E3P;
					REAL weno_P[8], weno_M[8];
					
					{
						//-------------		
						// z-dir-12
						//-------------
						REAL Ak1, Ak2, Ak3;
						REAL vs, E1, E2, E3, E1M, E2M, E3M;
						REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
						
						ss = sqrt(zStencil_Ax*zStencil_Ax + zStencil_Ay*zStencil_Ay + zStencil_Az*zStencil_Az); // 5
						REAL rcp = 1 / ss;
						Ak1 = zStencil_Ax *rcp;
						Ak2 = zStencil_Ay *rcp;
						Ak3 = zStencil_Az *rcp; // 10*3

						vs = zStencil_Ax * zStencil_u 
						   + zStencil_Ay * zStencil_v
						   + zStencil_Az * zStencil_w; // 5

						E1 = vs;
						E2 = vs - zStencil_cc * ss;
						E3 = vs + zStencil_cc * ss; // 4

						E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

						E1M = E1 - E1P;
						E2M = E2 - E2P;
						E3M = E3 - E3P; // 3
						// ----------------------------------------
						tmp0 = zStencil_d *Gamma_d_rcp; // 1
						uc1  = zStencil_u - zStencil_cc * Ak1; // 2
						uc2  = zStencil_u + zStencil_cc * Ak1; // 2
						
						REAL fp1,fm1,fp2,fm2;
						
						fp1 = tmp0 * (split_C1_d * E1P + E2P + E3P); // fpfm_1
						fm1 = tmp0 * (split_C1_d * E1M + E2M + E3M);
						fp2 = tmp0 * (split_C1_d * E1P * zStencil_u + E2P * uc1 + E3P * uc2); // fpfm_2
						fm2 = tmp0 * (split_C1_d * E1M * zStencil_u + E2M * uc1 + E3M * uc2);
						
						P_shared[8  +offset] = fp1; // fpfm_1
						M_shared[8  +offset] = fm1;
						P_shared[12 +offset] = fp2; // fpfm_2
						M_shared[12 +offset] = fm2;
						
						get_SoA_LAP(fpZ, x, y, z+4, 0) = fp1; // fpfm_1
						get_SoA_LAP(fmZ, x, y, z+4, 0) = fm1;
						get_SoA_LAP(fpZ, x, y, z+4, 1) = fp2; // fpfm_2
						get_SoA_LAP(fmZ, x, y, z+4, 1) = fm2;
						
								// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
									// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, tmp0 * (split_C1_d * E1M * zStencil_u + E2M * uc1 + E3M * uc2) ); // fpZ_buffer[0]
					}
					
							__syncthreads();
							// __threadfence();
							for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[1+offset +ii],
													  weno_M[ii] = M_shared[1+offset +ii];
					
					{
						//-------------		
						// z-dir-345
						//-------------
						REAL Ak1, Ak2, Ak3;
						REAL vs, E1, E2, E3, E1M, E2M, E3M;
						REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
						
						// ss = sqrt(zStencil_Ax*zStencil_Ax + zStencil_Ay*zStencil_Ay + zStencil_Az*zStencil_Az); // 5
						REAL rcp = 1 / ss;
						Ak1 = zStencil_Ax *rcp;
						Ak2 = zStencil_Ay *rcp;
						Ak3 = zStencil_Az *rcp; // 10*3

						vs = zStencil_Ax * zStencil_u 
						   + zStencil_Ay * zStencil_v
						   + zStencil_Az * zStencil_w; // 5

						E1 = vs;
						E2 = vs - zStencil_cc * ss;
						E3 = vs + zStencil_cc * ss; // 4

						// E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
						// E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50; // 15

						E1M = E1 - E1P;
						E2M = E2 - E2P;
						E3M = E3 - E3P; // 3
						// ----------------------------------------
						tmp0 = zStencil_d *Gamma_d_rcp; // 1
						uc1  = zStencil_u - zStencil_cc * Ak1;
						uc2  = zStencil_u + zStencil_cc * Ak1;
						vc1  = zStencil_v - zStencil_cc * Ak2;
						vc2  = zStencil_v + zStencil_cc * Ak2;
						wc1  = zStencil_w - zStencil_cc * Ak3;
						wc2  = zStencil_w + zStencil_cc * Ak3; // 12
						vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
						vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50; // 12
						vv = (Gamma_d - 1.0) * (zStencil_u * zStencil_u 
											  + zStencil_v * zStencil_v
											  + zStencil_w * zStencil_w ); // 7
						W2 = split_C3_d * zStencil_cc * zStencil_cc; // 2
						
						// P_shared[8  +offset] = tmp0 * (split_C1_d * E1P + E2P + E3P); // fpfm_1
						// M_shared[8  +offset] = tmp0 * (split_C1_d * E1M + E2M + E3M);
						// P_shared[12 +offset] = tmp0 * (split_C1_d * E1P * zStencil_u + E2P * uc1 + E3P * uc2); // fpfm_2
						// M_shared[12 +offset] = tmp0 * (split_C1_d * E1M * zStencil_u + E2M * uc1 + E3M * uc2);
						
						REAL fp3,fm3,fp4,fm4,fp5,fm5;
						
						fp3 = tmp0 * (split_C1_d * E1P * zStencil_v + E2P * vc1 + E3P * vc2); // fpfm_3
						fm3 = tmp0 * (split_C1_d * E1M * zStencil_v + E2M * vc1 + E3M * vc2);
						fp4 = tmp0 * (split_C1_d * E1P * zStencil_w + E2P * wc1 + E3P * wc2); // fpfm_4
						fm4 = tmp0 * (split_C1_d * E1M * zStencil_w + E2M * wc1 + E3M * wc2);
						fp5 = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P)); // fpfm_5
						fm5 = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
						
						P_shared[16 +offset] = fp3; // fpfm_3
						M_shared[16 +offset] = fm3;
						P_shared[20 +offset] = fp4; // fpfm_4
						M_shared[20 +offset] = fm4;
						P_shared[24 +offset] = fp5; // fpfm_5
						M_shared[24 +offset] = fm5;
						
						get_SoA_LAP(fpZ, x, y, z+4, 2) = fp3; // fpfm_3
						get_SoA_LAP(fmZ, x, y, z+4, 2) = fm3;
						get_SoA_LAP(fpZ, x, y, z+4, 3) = fp4; // fpfm_4
						get_SoA_LAP(fmZ, x, y, z+4, 3) = fm4;
						get_SoA_LAP(fpZ, x, y, z+4, 4) = fp5; // fpfm_5
						get_SoA_LAP(fmZ, x, y, z+4, 4) = fm5;
					}
					
							// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
								// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, P_shared[12 +offset]);
					
							// __syncthreads();
							P_shared[4+offset] = fpZ_buffer[1]; // fp_2 Left
							P_shared[8+offset] = fpZ_buffer[6]; // fp_2 center
					
					#pragma unroll 1
					for(int loop=1; loop<6; loop++)
					{
						int flag_faceZ = !threadIdx.y; // 1,0,0,0
						REAL weno_plus;
						
						{
							// weno+;
							REAL weno = OCFD_weno7_SYMBO_kernel_P_opt(&weno_P[0]);
							REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
							RHS[loop] += -Ajacobi*( weno - UP*!flag_faceZ )*hz_d_rcp;
							weno_plus = weno;
							
									// if(ZZZ == 1 && loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0 && threadIdx.x==0 && threadIdx.y==1 && threadIdx.z==0)
										// printf("YYY=%d, weno_P: %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E\n", YYY, weno_P[0],weno_P[1],weno_P[2],weno_P[3],weno_P[4],weno_P[5],weno_P[6],weno_P[7]);
									
									// if(ZZZ == 1 && loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
										// printf("YYY=%d, threadIdx.x=%d && threadIdx.y=%d && threadIdx.z=%d,\n weno_P: %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E\n", YYY, threadIdx.x,threadIdx.y,threadIdx.z, 
										// weno_P[0],weno_P[1],weno_P[2],weno_P[3],weno_P[4],weno_P[5],weno_P[6],weno_P[7]);
										
									// if(ZZZ == 1 && loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
										// printf("YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", YYY, threadIdx.x,threadIdx.y,threadIdx.z, weno);
									
									// if(loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
										// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, fpZ_buffer[0]); // fpZ_buffer[0]
						}
						
								if(loop<5)
								{
									M_shared[  loop*4+offset] = fmZ_buffer[loop], 
									M_shared[4+loop*4+offset] = fmZ_buffer[loop+5]; // WRITE: fm_2345*(左中)
									for(int ii=0; ii<8; ii++) weno_P[ii] = P_shared[1+loop*4+offset +ii]; // READ:  fp_2345*(左中右)
								}
					
						int offset_Previous = YYY*320 -63+loop*64 +threadIdx.x+16*threadIdx.z; // take x as the continuous direction in LDS. x-z: 16*4.
						REAL weno_Previous = FaceZ_shared[offset_Previous];
						
						{
							// weno-;
							REAL weno = OCFD_weno7_SYMBO_kernel_M_opt(&weno_M[0]);
							REAL UP   = __shfl_up_double(weno, 16, hipWarpSize);
							RHS[loop] += -Ajacobi*( weno - UP*!flag_faceZ )*hz_d_rcp;
							int flag_thread = threadIdx.y / 3; // 0,0,0,1
							FaceZ_shared[offset_Previous *flag_thread] = weno_plus + weno;
							
									// if(ZZZ == 1 && loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
										// printf("YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", YYY, threadIdx.x,threadIdx.y,threadIdx.z, weno);
									// if(loop == 1 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
										// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, weno);
									// if(loop == 1)
										// printf("ZZZ=%d, YYY=%d; blockIdx.x=%d, blockIdx.y=%d, blockIdx.z=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, blockIdx.x,blockIdx.y,blockIdx.z, threadIdx.x,threadIdx.y,threadIdx.z, weno);
									
									// if(loop == 5 && blockIdx.x==1 && blockIdx.y==1 && blockIdx.z==1)
										// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, weno);
						}
						RHS[loop] += Ajacobi *weno_Previous *hz_d_rcp *flag_faceZ;
						
								int flag_loop = loop/4; // 0,0,0,1,1
								if(loop<5)
								{
									P_shared[ (4+loop*4+offset)* !flag_loop] = fpZ_buffer[loop+1], 
									P_shared[ (8+loop*4+offset)* !flag_loop] = fpZ_buffer[loop+6]; // WRITE: fp_345**(左中)
									for(int ii=0; ii<8; ii++) weno_M[ii] = M_shared[1+loop*4+offset +ii]; // READ:  fm_2345*(左中右)
								}
							
							
									// if(loop == 4 && blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
										// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, RHS[loop] );
					}
				}
				
								// if(blockIdx.x==0 && blockIdx.y==0 && blockIdx.z==0)
									// printf("ZZZ=%d, YYY=%d; threadIdx.x=%d, threadIdx.y=%d, threadIdx.z=%d; weno_P: %24.16E\n", ZZZ, YYY, threadIdx.x,threadIdx.y,threadIdx.z, RHS[1] );
				
				unsigned int offset_tran = threadIdx.y+4*threadIdx.z+16*threadIdx.x;
				if( threadIdx.x != (blockDim.x-1) )
				{
					get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = RHS[1] +Trans_shared[      offset_tran];
					get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = RHS[2] +Trans_shared[256  +offset_tran];
					get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = RHS[3] +Trans_shared[512  +offset_tran];
					get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = RHS[4] +Trans_shared[768  +offset_tran];
					get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = RHS[5] +Trans_shared[1024 +offset_tran];
					
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = RHS[1] +rhs[1];
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = RHS[2] +rhs[2];
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = RHS[3] +rhs[3];
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = RHS[4] +rhs[4];
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = RHS[5] +rhs[5];
				}
			}
			
		} // Second line YYY loop.
		
	} // ZZZ Loop
}


#ifdef __cplusplus
}
#endif