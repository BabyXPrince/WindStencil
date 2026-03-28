#include "hip/hip_runtime.h"
#include <math.h>

#include "OCFD_NS_Jacobian3d.h"
#include "parameters.h"
#include "OCFD_Schemes_Choose.h"
#include "OCFD_split.h"

#include "commen_kernel.h"
#include "cuda_commen.h"
#include "cuda_utility.h"
#include "OCFD_mpi_dev.h"
#include "parameters_d.h"
#include "OCFD_flux_charteric.h"

#include "OCFD_warp_shuffle.h"
#include "OCFD_Schemes.h"
#include "OCFD_bound_Scheme.h"

#ifdef __cplusplus
extern "C" {
#endif

void du_invis_Jacobian3d_init(cudaJobPackage job_in, hipStream_t *stream){
	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);
	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x+2*LAP, size.y+2*LAP, size.z+2*LAP);
	
	cudaJobPackage job( dim3(job_in.start.x-LAP, job_in.start.y-LAP, job_in.start.z-LAP), 
						dim3(job_in.end.x + LAP, job_in.end.y + LAP, job_in.end.z + LAP) );
						
	CUDA_LAUNCH(( sound_speed_kernel<<<griddim , blockdim, 0, *stream>>>(*pT_d , *pcc_d , job) ));
}

/*
void du_invis_Jacobian3d_x(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, hipStream_t *stream){
	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);
	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x+2*LAP, size.y, size.z);

	cudaJobPackage job( dim3(job_in.start.x-LAP, job_in.start.y, job_in.start.z), 
	                    dim3(job_in.end.x + LAP, job_in.end.y, job_in.end.z) );
	
	CUDA_LAUNCH(( split_Jac3d_Stager_Warming_ker<<<griddim , blockdim, 0, *stream>>>( *pd_d, *pu_d, *pv_d, *pw_d, *pcc_d, *fp, *fm, *pAkx_d, *pAky_d, *pAkz_d, job) ));


	OCFD_dx1(*fp, *pdu_d, *pAjac_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAkx_d, *pAky_d, *pAkz_d, job_in, BlockDim_X, stream, D0_bound[0], D0_bound[1]);

	OCFD_dx2(*fm, *pdu_d, *pAjac_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAkx_d, *pAky_d, *pAkz_d, job_in, BlockDim_X, stream, D0_bound[0], D0_bound[1]);

}
*/


__global__ void split_Jac3d_Stager_Warming_ker_origin(cudaField d, cudaField u, cudaField v, cudaField w, cudaField cc, cudaSoA fp, cudaSoA fm,
	cudaField Ax, cudaField Ay, cudaField Az, cudaJobPackage job)
{
	// eyes on cells WITH LAPs

	unsigned int x = threadIdx.x + blockIdx.x*blockDim.x + job.start.x;
	unsigned int y = threadIdx.y + blockIdx.y*blockDim.y + job.start.y;
	unsigned int z = threadIdx.z + blockIdx.z*blockDim.z + job.start.z;

	REAL A1, A2, A3, Ak1, Ak2, Ak3, ss, tmp0;
	REAL E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
	REAL vs, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;

	if( x<job.end.x && y<job.end.y && z<job.end.z){
		A1 = get_Field_LAP(Ax, x, y, z);
		A2 = get_Field_LAP(Ay, x, y, z);
		A3 = get_Field_LAP(Az, x, y, z);

		ss = sqrt(A1*A1 + A2*A2 + A3*A3);
		Ak1 = A1 / ss;
		Ak2 = A2 / ss;
		Ak3 = A3 / ss;

		vs = A1 * get_Field_LAP(u, x, y, z) 
		   + A2 * get_Field_LAP(v, x, y, z) 
		   + A3 * get_Field_LAP(w, x, y, z);

		E1 = vs;
		E2 = vs - get_Field_LAP(cc, x, y, z) * ss;
		E3 = vs + get_Field_LAP(cc, x, y, z) * ss;

		E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
		E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
		E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

		E1M = E1 - E1P;
		E2M = E2 - E2P;
		E3M = E3 - E3P;
		// ----------------------------------------
		tmp0 = get_Field_LAP(d, x, y, z) / (2.0 * Gamma_d);
		uc1 = get_Field_LAP(u, x, y, z) - get_Field_LAP(cc, x, y, z) * Ak1;
		uc2 = get_Field_LAP(u, x, y, z) + get_Field_LAP(cc, x, y, z) * Ak1;
		vc1 = get_Field_LAP(v, x, y, z) - get_Field_LAP(cc, x, y, z) * Ak2;
		vc2 = get_Field_LAP(v, x, y, z) + get_Field_LAP(cc, x, y, z) * Ak2;
		wc1 = get_Field_LAP(w, x, y, z) - get_Field_LAP(cc, x, y, z) * Ak3;
		wc2 = get_Field_LAP(w, x, y, z) + get_Field_LAP(cc, x, y, z) * Ak3;
		vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
		vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50;
		vv = (Gamma_d - 1.0) * (get_Field_LAP(u, x, y, z) * get_Field_LAP(u, x, y, z) 
							  + get_Field_LAP(v, x, y, z) * get_Field_LAP(v, x, y, z) 
							  + get_Field_LAP(w, x, y, z) * get_Field_LAP(w, x, y, z) );
		W2 = split_C3_d * get_Field_LAP(cc, x, y, z) * get_Field_LAP(cc, x, y, z);
		
				// ---------------------------------
					// get_SoA_LAP(fp, x, y, z, 0) = get_Field_LAP(d, x, y, z);
					// get_SoA_LAP(fp, x, y, z, 1) = get_Field_LAP(u, x, y, z);
					// get_SoA_LAP(fp, x, y, z, 2) = get_Field_LAP(v, x, y, z);
					// get_SoA_LAP(fp, x, y, z, 3) = get_Field_LAP(w, x, y, z);
					// get_SoA_LAP(fp, x, y, z, 4) = get_Field_LAP(cc, x, y, z);
				// ---------------------------------

		get_SoA_LAP(fp, x, y, z, 0) = tmp0 * (split_C1_d * E1P + E2P + E3P);
		get_SoA_LAP(fp, x, y, z, 1) = tmp0 * (split_C1_d * E1P * get_Field_LAP(u, x, y, z) + E2P * uc1 + E3P * uc2);
		get_SoA_LAP(fp, x, y, z, 2) = tmp0 * (split_C1_d * E1P * get_Field_LAP(v, x, y, z) + E2P * vc1 + E3P * vc2);
		get_SoA_LAP(fp, x, y, z, 3) = tmp0 * (split_C1_d * E1P * get_Field_LAP(w, x, y, z) + E2P * wc1 + E3P * wc2);
		get_SoA_LAP(fp, x, y, z, 4) = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P));
		// --------------------------------------------------------

		get_SoA_LAP(fm, x, y, z, 0) = tmp0 * (split_C1_d * E1M + E2M + E3M);
		get_SoA_LAP(fm, x, y, z, 1) = tmp0 * (split_C1_d * E1M * get_Field_LAP(u, x, y, z) + E2M * uc1 + E3M * uc2);
		get_SoA_LAP(fm, x, y, z, 2) = tmp0 * (split_C1_d * E1M * get_Field_LAP(v, x, y, z) + E2M * vc1 + E3M * vc2);
		get_SoA_LAP(fm, x, y, z, 3) = tmp0 * (split_C1_d * E1M * get_Field_LAP(w, x, y, z) + E2M * wc1 + E3M * wc2);
		get_SoA_LAP(fm, x, y, z, 4) = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
	}
}


// void du_invis_Jacobian3d_x(cudaJobPackage job_in, hipStream_t *stream)
void du_invis_Jacobian3d_x(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, hipStream_t *stream)
{
	
		// /*
			{
				// Y-direction fpfm computation.
				// dim3 blockdim , griddim, size;
				// jobsize(&job_in, &size);
				// cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y+2*LAP, size.z);

				// cudaJobPackage job_originY( dim3(job_in.start.x, job_in.start.y-LAP, job_in.start.z) , 
									// dim3(job_in.end.x, job_in.end.y+LAP, job_in.end.z) );
				
				// CUDA_LAUNCH(( split_Jac3d_Stager_Warming_ker_origin<<<griddim , blockdim, 0, *stream>>>( *pd_d, *pu_d, *pv_d, *pw_d, *pcc_d, *fp, *fm, *pAix_d, *pAiy_d, *pAiz_d, job_originY) ));
				
				// printf("Y-direction fpfm computation is DONE.\n");
			}
			
			cudaSoA *pfp_dZ; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP] // by XX
			cudaSoA *pfm_dZ; // [5][nz-2*LAP][ny-2*LAP][nx-2*LAP] // by XX
			new_cudaSoA( &pfp_dZ , nx+2*LAP , ny+2*LAP , nz+2*LAP);
			new_cudaSoA( &pfm_dZ , nx+2*LAP , ny+2*LAP , nz+2*LAP);
			// {
				// dim3 blockdim , griddim, size;
				// jobsize(&job_in, &size);
				// cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z+2*LAP);

				// cudaJobPackage job_originZ( dim3(job_in.start.x, job_in.start.y, job_in.start.z-LAP) , 
									// dim3(job_in.end.x, job_in.end.y, job_in.end.z+LAP) );

				// CUDA_LAUNCH(( split_Jac3d_Stager_Warming_ker_origin<<<griddim , blockdim, 0, *stream>>>( *pd_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pfp_dZ, *pfm_dZ, *pAsx_d, *pAsy_d, *pAsz_d, job_originZ) ));
			// }
		// */
	
	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);
	cudaJobPackage job( dim3(job_in.start.x, job_in.start.y, job_in.start.z), 
	                    dim3(job_in.end.x  , job_in.end.y  , job_in.end.z) );
						
	printf("job.start.x=%d, job.end.x=%d, job.start.y=%d, job.end.y=%d, job.start.z=%d, job.end.z=%d.\n", job.start.x,job.end.x,job.start.y,job.end.y,job.start.z,job.end.z);
						
	// cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z); // without overlap.
	
			// cal_grid_block_dim(&griddim, &blockdim, BlockDimX-1, BlockDimY-1, BlockDimZ-1, size.x, size.y, size.z); // with xyz overlap.
            // blockdim.x+=1, blockdim.y+=1, blockdim.z+=1; // (16,4,4)的线程块 负责计算15.3.3的物理网格.

			// cal_grid_block_dim(&griddim, &blockdim, BlockDimX-1, BlockDimY, BlockDimZ, size.x, size.y, size.z); // with xy overlap.
			// blockdim.x+=1, blockdim.y+=1; // (16,5,4)的线程块 负责计算15.4.4的物理网格.

			#define scanY 8
			#define scanZ 16

			cal_grid_block_dim(&griddim, &blockdim, BlockDimX-1, BlockDimY, BlockDimZ, size.x, size.y, size.z); // without x overlap.
			blockdim.x+=1; // (16,4,4)的线程块，负责计算15.4.4的物理网格.
			griddim.x-=2, griddim.y-=2; // 扣掉x和y方向的两侧边界块.
			griddim.y /= scanY; // 扣掉x和y方向的两侧边界块.
			griddim.z /= scanZ; // 扣掉x和y方向的两侧边界块.
			// griddim.x=1, griddim.y=1, griddim.z=1;

	printf("inner: blockdim=%d,%d,%d, griddim=%d,%d,%d.\n", blockdim.x, blockdim.y, blockdim.z, griddim.x, griddim.y, griddim.z);
	// printf("HOST: epsl_sw_d = %f, split_C1_d = %f.\n", epsl_sw_d, split_C1_d);
	
		// /*
			// initiate following test variables.
			int szx=nx, szy=ny, szz=nz; // 内点.
			// int szx=nx+2*LAP, szy=ny+2*LAP, szz=nz+2*LAP; // x方向. x+1 is because overlap of x-direction.
			// int szx=nx+1, szy=ny+2*LAP, szz=nz; // x方向. x+1 is because overlap of x-direction.
			// int szx=nx+1, szy=ny+2*LAP, szz=nz; // y方向. x+1 is because overlap of x-direction.
			// int szx=nx+1, szy=ny+2*LAP, szz=nz; // z方向. x+1 is because overlap of x-direction.
			printf("szx=%d, szy=%d, szz=%d\n", szx, szy, szz);

			// cudaField *test_x;
			// cudaField *test_y;
			// cudaField *test_z;
			// cudaSoA *test_fp;
			// cudaSoA *test_fm;
			// new_cudaField( &test_x,  szx, szy, szz );
			// new_cudaField( &test_y,  szx, szy, szz );
			// new_cudaField( &test_z,  szx, szy, szz );
			// new_cudaSoA( &test_fp, szx, szy, szz*8 );
			// new_cudaSoA( &test_fm, szx, szy, szz*8 );
			// cuda_mem_value_init_warp(0.0, test_x->ptr,  test_x->pitch,  test_x->pitch, szy, szz );
			// cuda_mem_value_init_warp(0.0, test_y->ptr,  test_y->pitch,  test_x->pitch, szy, szz );
			// cuda_mem_value_init_warp(0.0, test_z->ptr,  test_z->pitch,  test_x->pitch, szy, szz );
			// cuda_mem_value_init_warp(0.0, test_fp->ptr, test_fp->pitch, test_x->pitch, szy, szz*8 );
			// cuda_mem_value_init_warp(0.0, test_fm->ptr, test_fm->pitch, test_x->pitch, szy, szz*8 );
			// printf("test_x->pitch=%d\n", test_x->pitch);
			// printf("pd_d->pitch=%d\n",      pd_d->pitch);
			// printf("pdu_d->pitch=%d, pdu_d->length_Y=%d, pdu_d->length_Z=%d\n",
							// pdu_d->pitch,    pdu_d->length_Y,        pdu_d->length_Z );
			
			double Gamma_d_host, hx_d_host, hy_d_host, hz_d_host;
			hipMemcpyFromSymbol(&Gamma_d_host, Gamma_d, sizeof(double));
			hipMemcpyFromSymbol(&hx_d_host, hx_d, sizeof(double));
			hipMemcpyFromSymbol(&hy_d_host, hy_d, sizeof(double));
			hipMemcpyFromSymbol(&hz_d_host, hz_d, sizeof(double));
			// printf("Gamma_d = %f, hx_d = %f, hy_d = %f, hz_d = %f.\n", Gamma_d_host, hx_d_host, hy_d_host, hz_d_host);
		// */
	
	hipEvent_t  st,ed;
	hipEventCreate(&st);
    hipEventCreate(&ed);
	float timer;
	hipEventRecord(st,*stream);
	hipEventSynchronize(st);
			
		// /*
			CUDA_LAUNCH(( split_Jac3d_Stager_Warming_ker<<<griddim , blockdim, 0, *stream>>>(
			*pd_d, *pu_d, *pv_d, *pw_d, *pcc_d,
			// *pT_d,
			*pAjac_d, *pdu_d,
			*fp, *fm, // test
			*pfp_dZ, *pfm_dZ, // test
			*pAkx_d, *pAky_d, *pAkz_d,
			*pAix_d, *pAiy_d, *pAiz_d,
			*pAsx_d, *pAsy_d, *pAsz_d,
			// flagxyzb1,flagxyzb2,flagxyzb3,
			// flagxyzb4,flagxyzb5,flagxyzb6,
			job, 0.5/Gamma_d_host, 1.0/hx_d_host, 1.0/hy_d_host, 1.0/hz_d_host
			// job, *test_x, *test_y, *test_z, *test_fp, *test_fm
			) ));
		// */

	hipEventRecord(ed,*stream);
	hipEventSynchronize(ed);
	hipEventElapsedTime(&timer,st,ed);
	printf("split_Jac3d_Stager_Warming_ker time=%f ms.\n",timer);
		
					delete_cudaSoA(pfp_dZ);
					delete_cudaSoA(pfm_dZ);
	
		// /*
			griddim.y *= scanY;
			griddim.z *= scanZ;
			printf("(original)griddim=%d,%d,%d.\n", griddim.x, griddim.y, griddim.z);
			
			dim3 griddim_bc;
			griddim_bc.x = 2*griddim.x + 2*griddim.y + 4; // x和y的四周的边界块.
			griddim_bc.y = 1;
			griddim_bc.z = griddim.z;			
			
			CUDA_LAUNCH(( split_Jac3d_Stager_Warming_ker_boundary<<<griddim_bc , blockdim, 0, *stream>>>(
			*pd_d, *pu_d, *pv_d, *pw_d, *pcc_d,
			// *pT_d,
			*pAjac_d, *pdu_d,
			*pAkx_d, *pAky_d, *pAkz_d,
			*pAix_d, *pAiy_d, *pAiz_d,
			*pAsx_d, *pAsy_d, *pAsz_d,
			griddim,
			job, 0.5/Gamma_d_host, 1.0/hx_d_host, 1.0/hy_d_host, 1.0/hz_d_host
			// job, *test_x, *test_y, *test_z, *test_fp, *test_fm
			) ));
		// */
	
	// hipEventRecord(ed,*stream);
	// hipEventSynchronize(ed);
	// hipEventElapsedTime(&timer,st,ed);
	// printf("split_Jac3d_Stager_Warming_ker time=%f ms.\n",timer);
	
					// delete_cudaSoA(pfp_dZ);
					// delete_cudaSoA(pfm_dZ);
	
	printf("boundary: blockdim=%d,%d,%d, griddim_bc=%d,%d,%d.\n", blockdim.x, blockdim.y, blockdim.z, griddim_bc.x, griddim_bc.y, griddim_bc.z); // boundary kernel.
	
	/*
		// test check.
			double *x_test =(double*)calloc( (szz*szy*(test_x->pitch)), sizeof(double) );
			double *y_test =(double*)calloc( (szz*szy*(test_x->pitch)), sizeof(double) );
			double *z_test =(double*)calloc( (szz*szy*(test_x->pitch)), sizeof(double) );
			double *fp =(double*)calloc( (8*szz*szy*(test_x->pitch)), sizeof(double) );
			double *fm =(double*)calloc( (8*szz*szy*(test_x->pitch)), sizeof(double) );
			
			// hipMemcpy(x_test, test_x->ptr, (szz)*(szy)*(test_x->pitch)*sizeof(double), hipMemcpyDeviceToHost);
			// hipMemcpy(y_test, test_y->ptr, (szz)*(szy)*(test_x->pitch)*sizeof(double), hipMemcpyDeviceToHost);
			// hipMemcpy(z_test, test_z->ptr, (szz)*(szy)*(test_x->pitch)*sizeof(double), hipMemcpyDeviceToHost);
			// hipMemcpy(fp, test_fp->ptr, 8*(szz)*(szy)*(test_x->pitch)*sizeof(double), hipMemcpyDeviceToHost);
			// hipMemcpy(fm, test_fm->ptr, 8*(szz)*(szy)*(test_x->pitch)*sizeof(double), hipMemcpyDeviceToHost);

			FILE *file1_index = fopen( "check_fp.txt" , "w");

				for (int k = 0; k < szz; k++) {
				for (int j = 4; j < szy-4; j++) {
				for (int i = 15; i < szx-15; i++) {
					
					int offset[8];
					for (int n = 0; n < 8; n++) {
						// x + SoA.pitch*( y + ny_d*(z+ (var)*nz_d) )
						offset[n] = i + (test_x->pitch)*( j + szy*(k + n*szz) );
					}
					
					// fprintf(file1_index , "%d, %d, %d, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E\n",
										// i, j, k, fp[offset[0]], fp[offset[1]], fp[offset[2]], fp[offset[3]], fp[offset[4]], fp[offset[5]], fp[offset[6]], fp[offset[7]] ); // 8 parameters.
					
					// fprintf(file1_index , "%d, %d, %d, %24.16E\n",
										// i, j, k, fp[offset[2]]); // 1 parameters.
				}
				}
				}
			fclose(file1_index);
				
				// for (int k = 4; k < szz-LAP; k++) {
				// for (int j = 4; j < szy-LAP; j++) {
				// for (int i = 4; i < szx-LAP; i++) {
					
				// for (int k = 4; k < szz-LAP; k++) {
				// for (int j = 4; j < szy-LAP; j++) {
				// for (int i = 0; i < szx    ; i++) {
				// for (int i = 225; i < 248; i++) { // X-direction

				// for (int k = 4; k < szz-LAP; k++) {
				// for (int j = 0; j < szy    ; j++) { // Y-direction
				// for (int j = 60; j < 64; j++) {
				// for (int i = 4; i < szx-LAP; i++) {

				// for (int k = 0; k < szz    ; k++) { // Z-direction
				// for (int j = 4; j < szy-LAP; j++) {
				// for (int i = 4; i < szx-LAP; i++) {

					// int offset = i + j* (test_x->pitch) + k *szy*(test_x->pitch);

					// fprintf(file1_index , "%d, %d, %d, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E\n",
											// i-4, j-4, k-4, x_test[offset], y_test[offset], z_test[offset], fp[offset], fm[offset] ); // 5 parameters.

					// fprintf(file1_index , "%d, %d, %d, %24.16E, %24.16E, %24.16E\n",
											// i-4, j-4, k-4, x_test[offset], y_test[offset], z_test[offset] ); // 3 parameters.
					
					// fprintf(file1_index , "%d, %d, %d, %24.16E, %24.16E\n", i-4, j-4, k-4, x_test[offset], y_test[offset] ); // 2 parameters. (rhs)
					// fprintf(file1_index , "%d, %d, %d, %24.16E, %24.16E\n", i, j, k, x_test[offset], y_test[offset] ); // 2 parameters. (fp&fm)
			
					// fprintf(file1_index , "%d, %d, %d, %24.16E\n", i-4, j-4, k-4, x_test[offset] ); // 1 parameter. (rhs)
					// fprintf(file1_index , "%d, %d, %d, %24.16E\n", i, j, k, x_test[offset] ); // 1 parameter. (fp&fm)
				// }
				// }
				// }
				// fclose(file1_index);
			
			
			free(x_test), free(y_test), free(z_test), free(fp), free(fm);
			delete_cudaField(test_x);
			delete_cudaField(test_y);
			delete_cudaField(test_z);
			delete_cudaSoA(test_fp);
			delete_cudaSoA(test_fm);
	*/
	
	/*
		// rhs1 ~ rhs5.
		
		// if(tt > end_time-dt)
		if( tt - (end_time-dt) > 1e-24 || tt == (end_time-dt) )
		{
			printf("dt = %24.16E\n", dt);
			printf("tt = %24.16E\n", tt);
			printf("end_time = %24.16E\n", end_time);
			double *rhs_test =(double*)calloc( (5*nz*ny*(pdu_d->pitch)), sizeof(double) );
			hipMemcpy(rhs_test, pdu_d->ptr, 5*nz*ny*(pdu_d->pitch)*sizeof(double), hipMemcpyDeviceToHost);
			FILE *file2_index = fopen( "RHS_zhang.txt" , "w");
			double Sigma = 0.0;

					for (int k = 0; k < nz; k++) {
					for (int j = 0; j < ny; j++) {
					for (int i = 0; i < nx; i++) {

							// int offset = i + (pdu_d->pitch)*(j+ny*(k+var*nz));
							int offset1 = i + j* (pdu_d->pitch) + k *ny*(pdu_d->pitch);
							int offset2 = (pdu_d->pitch)*ny*nz;
							
							Sigma += rhs_test[offset1+offset2*0] +rhs_test[offset1+offset2*1] +rhs_test[offset1+offset2*2] +rhs_test[offset1+offset2*3] +rhs_test[offset1+offset2*4];

							fprintf(file2_index , "%d, %d, %d, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E\n",
												i, j, k, rhs_test[offset1+offset2*0], rhs_test[offset1+offset2*1], rhs_test[offset1+offset2*2], rhs_test[offset1+offset2*3], rhs_test[offset1+offset2*4] );
							// fprintf(file2_index , "%d, %d, %d, %24.16E\n",
																	// i, j, k, rhs_test[offset1+offset2*4] );
					}
					}
					}
			fclose(file2_index);
			free(rhs_test);
			printf("Sigma = %24.16E\n", Sigma);
		}
	*/
}

/*
void du_invis_Jacobian3d_y(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, hipStream_t *stream){
	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);
	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y+2*LAP, size.z);

	cudaJobPackage job( dim3(job_in.start.x, job_in.start.y-LAP, job_in.start.z) , 
	                    dim3(job_in.end.x, job_in.end.y+LAP, job_in.end.z) );
	
	CUDA_LAUNCH(( split_Jac3d_Stager_Warming_ker<<<griddim , blockdim, 0, *stream>>>( *pd_d, *pu_d, *pv_d, *pw_d, *pcc_d, *fp, *fm, *pAix_d, *pAiy_d, *pAiz_d, job) ));


	OCFD_dy1(*fp, *pdu_d, *pAjac_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAix_d, *pAiy_d, *pAiz_d, job_in, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);

	OCFD_dy2(*fm, *pdu_d, *pAjac_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAix_d, *pAiy_d, *pAiz_d, job_in, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);

}


void du_invis_Jacobian3d_z(cudaJobPackage job_in, cudaSoA *fp, cudaSoA *fm, hipStream_t *stream){
	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);
	cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , size.x , size.y , size.z+2*LAP);

	cudaJobPackage job( dim3(job_in.start.x, job_in.start.y, job_in.start.z-LAP) , 
	                    dim3(job_in.end.x, job_in.end.y, job_in.end.z+LAP) );

	CUDA_LAUNCH(( split_Jac3d_Stager_Warming_ker<<<griddim , blockdim, 0, *stream>>>( *pd_d, *pu_d, *pv_d, *pw_d, *pcc_d, *fp, *fm, *pAsx_d, *pAsy_d, *pAsz_d, job) ));


	OCFD_dz1(*fp, *pdu_d, *pAjac_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAsx_d, *pAsy_d, *pAsz_d, job_in, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);

	OCFD_dz2(*fm, *pdu_d, *pAjac_d, *pu_d, *pv_d, *pw_d, *pcc_d, *pAsx_d, *pAsy_d, *pAsz_d, job_in, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);

}
*/

// ========================================================

void du_viscous_Jacobian3d_init(hipStream_t *stream){

	cudaJobPackage job( dim3(LAP, LAP, LAP) , dim3(nx_lap, ny_lap, nz_lap) );

    OCFD_dx0(*pu_d, *puk_d, job, BlockDim_X, stream, D0_bound[0], D0_bound[1]);
    OCFD_dx0(*pv_d, *pvk_d, job, BlockDim_X, stream, D0_bound[0], D0_bound[1]);
    OCFD_dx0(*pw_d, *pwk_d, job, BlockDim_X, stream, D0_bound[0], D0_bound[1]);
	OCFD_dx0(*pT_d, *pTk_d, job, BlockDim_X, stream, D0_bound[0], D0_bound[1]);
	
    OCFD_dy0(*pu_d, *pui_d, job, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);
    OCFD_dy0(*pv_d, *pvi_d, job, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);
    OCFD_dy0(*pw_d, *pwi_d, job, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);
	OCFD_dy0(*pT_d, *pTi_d, job, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);
	
    OCFD_dz0(*pu_d, *pus_d, job, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);
    OCFD_dz0(*pv_d, *pvs_d, job, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);
    OCFD_dz0(*pw_d, *pws_d, job, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);
	OCFD_dz0(*pT_d, *pTs_d, job, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);

}


__global__ void vis_flux_ker(
cudaField uk,
cudaField vk,
cudaField wk,
cudaField ui,
cudaField vi,
cudaField wi,
cudaField us,
cudaField vs,
cudaField ws,

cudaField Tk,
cudaField Ti,
cudaField Ts,

cudaField Amu,

cudaField u,
cudaField v,
cudaField w,

cudaField Ax,
cudaField Ay,
cudaField Az,

cudaField Ajac,
cudaField Akx,
cudaField Aky,
cudaField Akz,
cudaField Aix,
cudaField Aiy,
cudaField Aiz,
cudaField Asx,
cudaField Asy,
cudaField Asz,

cudaField Ev1,
cudaField Ev2,
cudaField Ev3,
cudaField Ev4,
cudaJobPackage job)
{
	// eyes on cells WITH LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;


    if( x<job.end.x && y<job.end.y && z<job.end.z){

		REAL s11, s12, s13, s22, s23, s33;
		REAL E1, E2, E3;
		{
            REAL ux , vx , wx;
		    REAL uy , vy , wy;
		    REAL uz , vz , wz;
		    REAL div , Amu1;
    
		    ux=get_Field(uk, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Akx, x, y, z)
		      +get_Field(ui, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aix, x, y, z)
		      +get_Field(us, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Asx, x, y, z);
		    vx=get_Field(vk, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Akx, x, y, z)
		      +get_Field(vi, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aix, x, y, z)
		      +get_Field(vs, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Asx, x, y, z);
		    wx=get_Field(wk, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Akx, x, y, z)
		      +get_Field(wi, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aix, x, y, z)
		      +get_Field(ws, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Asx, x, y, z);
    
		    uy=get_Field(uk, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aky, x, y, z)
		      +get_Field(ui, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aiy, x, y, z)
		      +get_Field(us, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Asy, x, y, z);
		    vy=get_Field(vk, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aky, x, y, z)
		      +get_Field(vi, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aiy, x, y, z)
		      +get_Field(vs, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Asy, x, y, z);
		    wy=get_Field(wk, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aky, x, y, z)
		      +get_Field(wi, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aiy, x, y, z)
		      +get_Field(ws, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Asy, x, y, z);
		    
		    uz=get_Field(uk, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Akz, x, y, z)
		      +get_Field(ui, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aiz, x, y, z)
		      +get_Field(us, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Asz, x, y, z);
		    vz=get_Field(vk, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Akz, x, y, z)
		      +get_Field(vi, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aiz, x, y, z)
		      +get_Field(vs, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Asz, x, y, z);
		    wz=get_Field(wk, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Akz, x, y, z)
		      +get_Field(wi, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aiz, x, y, z)
		      +get_Field(ws, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Asz, x, y, z);
    
		    div=ux+vy+wz;
    
		    Amu1=get_Field(Amu, x-LAP, y-LAP, z-LAP);
		    	
		    s11 = (2.0*ux-2.0/3.0*div) * Amu1;
		    s22 = (2.0*vy-2.0/3.0*div) * Amu1;
		    s33 = (2.0*wz-2.0/3.0*div) * Amu1;
    
		    s12 = (uy+vx)*Amu1;
		    s13 = (uz+wx)*Amu1;
			s23 = (vz+wy)*Amu1;
		}

		{
			REAL Tx;
			REAL Ty;
			REAL Tz;
			REAL Amuk;
	
			Amuk=get_Field(Amu, x-LAP, y-LAP, z-LAP) * vis_flux_init_c_d;
				
			Tx=get_Field(Tk, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Akx, x, y, z)
			  +get_Field(Ti, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aix, x, y, z)
			  +get_Field(Ts, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Asx, x, y, z);	
			Ty=get_Field(Tk, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aky, x, y, z)
			  +get_Field(Ti, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aiy, x, y, z)
			  +get_Field(Ts, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Asy, x, y, z);	
			Tz=get_Field(Tk, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Akz, x, y, z)
			  +get_Field(Ti, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Aiz, x, y, z)
			  +get_Field(Ts, x-LAP, y-LAP, z-LAP) * get_Field_LAP(Asz, x, y, z);
	
			E1=get_Field_LAP(u, x, y, z) * s11 
 			  +get_Field_LAP(v, x, y, z) * s12 
 			  +get_Field_LAP(w, x, y, z) * s13 + Amuk*Tx;
			E2=get_Field_LAP(u, x, y, z) * s12 
 			  +get_Field_LAP(v, x, y, z) * s22 
 			  +get_Field_LAP(w, x, y, z) * s23 + Amuk*Ty;
			E3=get_Field_LAP(u, x, y, z) * s13 
 			  +get_Field_LAP(v, x, y, z) * s23 
 			  +get_Field_LAP(w, x, y, z) * s33 + Amuk*Tz;
		}

		{	
		    REAL akx , aky , akz;
		    {
		    	REAL Aj1;
		    	Aj1=get_Field_LAP(Ajac , x,y,z);
    
		    	akx = get_Field_LAP(Ax, x, y, z)*Aj1;
		    	aky = get_Field_LAP(Ay, x, y, z)*Aj1;
		    	akz = get_Field_LAP(Az, x, y, z)*Aj1;
		    }
		    
		    get_Field_LAP(Ev1 , x,y,z) = ( akx*s11 + aky*s12 + akz*s13 );
		    get_Field_LAP(Ev2 , x,y,z) = ( akx*s12 + aky*s22 + akz*s23 ); 
		    get_Field_LAP(Ev3 , x,y,z) = ( akx*s13 + aky*s23 + akz*s33 );
			get_Field_LAP(Ev4 , x,y,z) = ( akx*E1  + aky*E2  + akz*E3  );
	    }
	}
}

void du_viscous_Jacobian3d_x_init(hipStream_t *stream){

	dim3 blockdim , griddim;
	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, nx, ny, nz);

	cudaJobPackage job( dim3(LAP, LAP, LAP) , dim3(nx_lap, ny_lap, nz_lap) );

	CUDA_LAUNCH(( vis_flux_ker<<<griddim , blockdim, 0, *stream>>>(*puk_d,*pvk_d,*pwk_d,*pui_d,*pvi_d,*pwi_d,*pus_d,*pvs_d,*pws_d,
		                                                           *pTk_d,*pTi_d,*pTs_d,*pAmu_d,
		                                                           *pu_d,*pv_d,*pw_d,*pAkx_d,*pAky_d,*pAkz_d,
																   *pAjac_d,*pAkx_d,*pAky_d,*pAkz_d,*pAix_d,*pAiy_d,*pAiz_d,*pAsx_d,*pAsy_d,*pAsz_d,
																   *pEv1_d,*pEv2_d,*pEv3_d,*pEv4_d,job) ));
}

void du_viscous_Jacobian3d_x_final(cudaJobPackage job_in, hipStream_t *stream){

	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);

	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

    OCFD_dx0(*pEv1_d, *vis_u_d, job_in, BlockDim_X, stream, D0_bound[0], D0_bound[1]);
    OCFD_dx0(*pEv2_d, *vis_v_d, job_in, BlockDim_X, stream, D0_bound[0], D0_bound[1]);
    OCFD_dx0(*pEv3_d, *vis_w_d, job_in, BlockDim_X, stream, D0_bound[0], D0_bound[1]);
    OCFD_dx0(*pEv4_d, *vis_T_d, job_in, BlockDim_X, stream, D0_bound[0], D0_bound[1]);

	cudaJobPackage job(dim3(job_in.start.x-LAP, job_in.start.y-LAP, job_in.start.z-LAP) , 
	                   dim3(job_in.end.x - LAP, job_in.end.y - LAP, job_in.end.z - LAP));

	int size_du = pdu_d->pitch*ny*nz;
	cudaField tmp_du;
	tmp_du.pitch = pdu_d->pitch;

	tmp_du.ptr = pdu_d->ptr + size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_u_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_v_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_w_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_T_d, *pAjac_d, job) ));
}

void du_viscous_Jacobian3d_y_init(hipStream_t *stream){

	dim3 blockdim , griddim;
	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, nx, ny, nz);

	cudaJobPackage job( dim3(LAP, LAP, LAP) , dim3(nx_lap, ny_lap, nz_lap) );

	CUDA_LAUNCH(( vis_flux_ker<<<griddim , blockdim, 0, *stream>>>(*puk_d,*pvk_d,*pwk_d,*pui_d,*pvi_d,*pwi_d,*pus_d,*pvs_d,*pws_d,
		                                                           *pTk_d,*pTi_d,*pTs_d,*pAmu_d,
		                                                           *pu_d,*pv_d,*pw_d,*pAix_d,*pAiy_d,*pAiz_d,
		                                                           *pAjac_d,*pAkx_d,*pAky_d,*pAkz_d,*pAix_d,*pAiy_d,*pAiz_d,*pAsx_d,*pAsy_d,*pAsz_d,
		                                                           *pEv1_d,*pEv2_d,*pEv3_d,*pEv4_d,job) ));
}

void du_viscous_Jacobian3d_y_final(cudaJobPackage job_in, hipStream_t *stream){

	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);

	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

    OCFD_dy0(*pEv1_d, *vis_u_d, job_in, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);
    OCFD_dy0(*pEv2_d, *vis_v_d, job_in, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);
    OCFD_dy0(*pEv3_d, *vis_w_d, job_in, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);
	OCFD_dy0(*pEv4_d, *vis_T_d, job_in, BlockDim_Y, stream, D0_bound[2], D0_bound[3]);

	cudaJobPackage job(dim3(job_in.start.x-LAP, job_in.start.y-LAP, job_in.start.z-LAP) , 
					   dim3(job_in.end.x - LAP, job_in.end.y - LAP, job_in.end.z - LAP));
					   
	int size_du = pdu_d->pitch*ny*nz;
	cudaField tmp_du;
	tmp_du.pitch = pdu_d->pitch;

	tmp_du.ptr = pdu_d->ptr + size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_u_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_v_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_w_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_T_d, *pAjac_d, job) ));
}


void du_viscous_Jacobian3d_z_init(hipStream_t *stream){

	dim3 blockdim , griddim;
	cal_grid_block_dim(&griddim , &blockdim , BlockDimX , BlockDimY , BlockDimZ , nx , ny , nz);

	cudaJobPackage job( dim3(LAP, LAP, LAP) , dim3(nx_lap, ny_lap, nz_lap) );

	CUDA_LAUNCH(( vis_flux_ker<<<griddim , blockdim, 0, *stream>>>(*puk_d,*pvk_d,*pwk_d,*pui_d,*pvi_d,*pwi_d,*pus_d,*pvs_d,*pws_d,
		                                                           *pTk_d,*pTi_d,*pTs_d,*pAmu_d,
		                                                           *pu_d,*pv_d,*pw_d,*pAsx_d,*pAsy_d,*pAsz_d,
	                                                        	   *pAjac_d,*pAkx_d,*pAky_d,*pAkz_d,*pAix_d,*pAiy_d,*pAiz_d,*pAsx_d,*pAsy_d,*pAsz_d,
		                                                           *pEv1_d,*pEv2_d,*pEv3_d,*pEv4_d,job) ));
}


void du_viscous_Jacobian3d_z_final(cudaJobPackage job_in, hipStream_t *stream){

	dim3 blockdim , griddim, size;
	jobsize(&job_in, &size);

	cal_grid_block_dim(&griddim, &blockdim, BlockDimX, BlockDimY, BlockDimZ, size.x, size.y, size.z);

	OCFD_dz0(*pEv1_d, *vis_u_d, job_in, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);
    OCFD_dz0(*pEv2_d, *vis_v_d, job_in, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);
    OCFD_dz0(*pEv3_d, *vis_w_d, job_in, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);
	OCFD_dz0(*pEv4_d, *vis_T_d, job_in, BlockDim_Z, stream, D0_bound[4], D0_bound[5]);

	cudaJobPackage job(dim3(job_in.start.x-LAP, job_in.start.y-LAP, job_in.start.z-LAP) , 
					   dim3(job_in.end.x - LAP, job_in.end.y - LAP, job_in.end.z - LAP));
					   
	int size_du = pdu_d->pitch*ny*nz;
	cudaField tmp_du;
	tmp_du.pitch = pdu_d->pitch;

	tmp_du.ptr = pdu_d->ptr + size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_u_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_v_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_w_d, *pAjac_d, job) ));

	tmp_du.ptr += size_du;
	CUDA_LAUNCH(( YF_Pe_XF<<<griddim , blockdim, 0, *stream>>>(tmp_du, *vis_T_d, *pAjac_d, job) ));
}

__global__ void boundary_symmetry_pole_vis_y_ker_m(
	cudaField Ev1,
	cudaField Ev2,
	cudaField Ev3,
	cudaField Ev4,
	cudaJobPackage job){

	// eyes on Bottom holo cells WITH LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if( x<job.end.x && y<job.end.y && z<job.end.z){
		unsigned int y1 = 2*LAP - y;

		get_Field_LAP(Ev1 , x,y,z) = - get_Field_LAP(Ev1 , x,y1,z);
		get_Field_LAP(Ev2 , x,y,z) =   get_Field_LAP(Ev2 , x,y1,z);
		get_Field_LAP(Ev3 , x,y,z) = - get_Field_LAP(Ev3 , x,y1,z);
		get_Field_LAP(Ev4 , x,y,z) = - get_Field_LAP(Ev4 , x,y1,z);
	}
}

__global__ void boundary_symmetry_pole_vis_y_ker_p(
	cudaField Ev1,
	cudaField Ev2,
	cudaField Ev3,
	cudaField Ev4,
	cudaJobPackage job){

	// eyes on Top holo cells WITH LAPs
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x + job.start.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y + job.start.y;
	unsigned int z = blockDim.z * blockIdx.z + threadIdx.z + job.start.z;

	if( x<job.end.x && y<job.end.y && z<job.end.z){
		unsigned int y1 = 2*(ny_d+LAP-1) - y;

		get_Field_LAP(Ev1 , x,y,z) = - get_Field_LAP(Ev1 , x,y1,z);
		get_Field_LAP(Ev2 , x,y,z) =   get_Field_LAP(Ev2 , x,y1,z);
		get_Field_LAP(Ev3 , x,y,z) = - get_Field_LAP(Ev3 , x,y1,z);
		get_Field_LAP(Ev4 , x,y,z) = - get_Field_LAP(Ev4 , x,y1,z);
	}
}

void boundary_symmetry_pole_vis_y(hipStream_t *stream){
	dim3 blockdim , griddim;
//    symmetry or pole boundary condition for viscous term
    if(IF_SYMMETRY == 1){
        if(npy == 0){
		    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , LAP , BlockDimZ , nx , LAP , nz);
		    cudaJobPackage job(dim3(LAP , 0 , LAP) , dim3(nx_lap , LAP , nz_lap));
		    CUDA_LAUNCH(( boundary_symmetry_pole_vis_y_ker_m<<<griddim , blockdim, 0, *stream>>>(*pEv1_d,*pEv2_d,*pEv3_d,*pEv4_d , job) ));
	    }
	    if(npy == NPY0-1){
		    cal_grid_block_dim(&griddim , &blockdim , BlockDimX , LAP , BlockDimZ , nx , LAP , nz);
		    cudaJobPackage job(dim3(LAP , ny_lap , LAP) , dim3(nx_lap , ny_2lap , nz_lap));
		    CUDA_LAUNCH(( boundary_symmetry_pole_vis_y_ker_p<<<griddim , blockdim, 0, *stream>>>(*pEv1_d,*pEv2_d,*pEv3_d,*pEv4_d , job) ));
    	}
    }
}


__global__ void split_Jac3d_Stager_Warming_ker_boundary(cudaField d, cudaField u, cudaField v, cudaField w, 
		cudaField cc, 
		// cudaField T,
		cudaField Ajac , cudaSoA du,
		cudaField Ax, cudaField Ay, cudaField Az,cudaField Bx, cudaField By, cudaField Bz,
		cudaField Cx, cudaField Cy, cudaField Cz, 
		dim3 griddim_Inner,
		cudaJobPackage job, double Gamma_d_rcp, double hx_d_rcp, double hy_d_rcp, double hz_d_rcp
		// cudaJobPackage job, cudaField test_x, cudaField test_y, cudaField test_z, cudaField test_fp, cudaField test_fm
	)
{
	// eyes on cells WITH LAPs
	// unsigned int x = threadIdx.x + blockIdx.x*blockDim.x + job.start.x;
	// unsigned int y = threadIdx.y + blockIdx.y*blockDim.y + job.start.y;
	// unsigned int z = threadIdx.z + blockIdx.z*blockDim.z + job.start.z;	// without overlalp
	
	//  ****************************
	//  *         x_start          *
	//  *--------------------------*
	//  *     |              |     *		y
	//  *     |              |     *		^
	//  * y_s |              | y_e *		|
	//  *     |              |     *		|
	//  *     |              |     *		---->x
	//  *--------------------------*
	//  *          x_end           *
	//  ****************************
	
	unsigned int x, y, z;
	
	// x_start
	if( blockIdx.x < griddim_Inner.x+2 ){
		x = threadIdx.x + blockIdx.x*(blockDim.x-1) + job.start.x;
		y = threadIdx.y + job.start.y;
		z = threadIdx.z + blockIdx.z*blockDim.z + job.start.z;
	}
	
	// x_end
	if( blockIdx.x >= griddim_Inner.x+2 && blockIdx.x < 2*(griddim_Inner.x+2) ){
		x = threadIdx.x + (blockIdx.x-griddim_Inner.x-2)*(blockDim.x-1) + job.start.x;
		y = threadIdx.y + (griddim_Inner.y+1)*blockDim.y + job.start.y;
		z = threadIdx.z + blockIdx.z*blockDim.z + job.start.z;
	}
	
	// y_start
	if( blockIdx.x >= 2*(griddim_Inner.x+2) && blockIdx.x < 2*(griddim_Inner.x+2)+ griddim_Inner.y ){
		x = threadIdx.x + job.start.x;
		y = threadIdx.y + (blockIdx.x-2*griddim_Inner.x-3)*blockDim.y + job.start.y; // -3 = -2*2+1
		z = threadIdx.z + blockIdx.z*blockDim.z + job.start.z;
	}
	
	// y_end
	if( blockIdx.x >= 2*(griddim_Inner.x+2)+ griddim_Inner.y ){
		x = threadIdx.x + (griddim_Inner.x+1)*(blockDim.x-1) + job.start.x;
		y = threadIdx.y + (blockIdx.x-2*griddim_Inner.x-griddim_Inner.y-3)*blockDim.y + job.start.y;
		z = threadIdx.z + blockIdx.z*blockDim.z + job.start.z;
	}
	
	REAL stencil_d[3];
	REAL stencil_u[3];
	REAL stencil_v[3];
	REAL stencil_w[3];
	REAL stencil_cc[3];
	
	REAL stencil_Ax[3];
	REAL stencil_Ay[3];
	REAL stencil_Az[3];
	
	REAL Ajacobi;
	REAL rhs1 = 0, rhs2 = 0, rhs3 = 0, rhs4 = 0, rhs5 = 0; // rhs第一步，需要初始化为0，很重要.
	
	__shared__ REAL  d_shared[257]; // 16*4*4+1
	__shared__ REAL  u_shared[257];
	__shared__ REAL  v_shared[257];
	__shared__ REAL  w_shared[257];
	__shared__ REAL cc_shared[257];
	
	__shared__ REAL fp_shared[768];
	__shared__ REAL fm_shared[768];
	
					// ***********************************
						// get_Field_LAP(test_x, x, y, z) = x - job.start.x;
						// get_Field_LAP(test_y, x, y, z) = y - job.start.y;
						// get_Field_LAP(test_z, x, y, z) = z - job.start.z;
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
			
			//
			//for rho parameters
			stencil_d[0] = get_Field_LAP(d, x-4, y, z) ;
			stencil_d[1] = get_Field_LAP(d, x+3, y, z) ;
			
				//store inner flow data to lds buffer
				d_shared[ ( 1 +threadIdx.x-4 +16*threadIdx.y +64*threadIdx.z) *flag1] = stencil_d[0];
				d_shared[ ( 1 +threadIdx.x+3 +16*threadIdx.y +64*threadIdx.z) *flag2] = stencil_d[1];
			
			//
			//for u parameters
			stencil_u[0] = get_Field_LAP(u, x-4, y, z) ;
			stencil_u[1] = get_Field_LAP(u, x+3, y, z) ;
				
				//store inner flow data to lds buffer 
				u_shared[ ( 1 +threadIdx.x-4 +16*threadIdx.y +64*threadIdx.z) *flag1] = stencil_u[0];
				u_shared[ ( 1 +threadIdx.x+3 +16*threadIdx.y +64*threadIdx.z) *flag2] = stencil_u[1];
			
			//
			//for v parameters
			stencil_v[0] = get_Field_LAP(v, x-4, y, z) ;
			stencil_v[1] = get_Field_LAP(v, x+3, y, z) ;
			
				//store inner flow data to lds buffer 
				v_shared[ ( 1 +threadIdx.x-4 +16*threadIdx.y +64*threadIdx.z) *flag1] = stencil_v[0];
				v_shared[ ( 1 +threadIdx.x+3 +16*threadIdx.y +64*threadIdx.z) *flag2] = stencil_v[1];
			
			//
			//for w parameters
			stencil_w[0] = get_Field_LAP(w, x-4, y, z) ;
			stencil_w[1] = get_Field_LAP(w, x+3, y, z) ;
				
				//store inner flow data to lds buffer 
				w_shared[ ( 1 +threadIdx.x-4 +16*threadIdx.y +64*threadIdx.z) *flag1] = stencil_w[0];
				w_shared[ ( 1 +threadIdx.x+3 +16*threadIdx.y +64*threadIdx.z) *flag2] = stencil_w[1];
			
			//
			//for T/cc parameters
			stencil_cc[0] = get_Field_LAP(cc, x-4, y, z);
			stencil_cc[1] = get_Field_LAP(cc, x+3, y, z);
				
				//store inner flow data to lds buffer 
				cc_shared[ ( 1 +threadIdx.x-4 +16*threadIdx.y +64*threadIdx.z) *flag1] = stencil_cc[0];
				cc_shared[ ( 1 +threadIdx.x+3 +16*threadIdx.y +64*threadIdx.z) *flag2] = stencil_cc[1];
			
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
		
		//computing the flux+ and flux -
		//store in stencil_fm and stencil_fp		
		// if( x>=job.start.x && x<=job.end.x && y>=job.start.y && y<=job.end.y && z>=job.start.z && z<=job.end.z)
		{
			// int offset = threadIdx.x+24*threadIdx.y+96*threadIdx.z; // take x as the continuous direction in LDS. x-y-z: 24*4*4.
			int offset = threadIdx.x+23*threadIdx.y+92*threadIdx.z; // take x as the continuous direction in LDS. x-y-z: 23*4*4.
			
			//---------------------
			// computing rhs1
			//---------------------
			for(int ii=0; ii<2; ii++)
			{
				//-------------
				// x-dir
				//-------------
				REAL ss;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				
				// take z as the continuous direction in LDS. z-y-x: 12*4*16
				// fp_shared[8*ii +offset] = tmp0 * (split_C1_d * E1P + E2P + E3P);
				// fm_shared[8*ii +offset] = tmp0 * (split_C1_d * E1M + E2M + E3M);
				fp_shared[7*ii +offset] = tmp0 * (split_C1_d * E1P + E2P + E3P);
				fm_shared[7*ii +offset] = tmp0 * (split_C1_d * E1M + E2M + E3M);
			}
			
			{
				// REAL rhs = 0; // rhs1第一步，需要初始化为0，很重要.
				// int flag;
				
					// **************************************************
							// get_Field_LAP(test_x, x-4, y, z-4) = fp_shared[threadIdx.x   + 24*threadIdx.y + 96*threadIdx.z];
							// get_Field_LAP(test_y, x-4, y, z-4) = fp_shared[threadIdx.x+1 + 24*threadIdx.y + 96*threadIdx.z];
							// get_Field_LAP(test_z, x-4, y, z-4) = fp_shared[threadIdx.x+2 + 24*threadIdx.y + 96*threadIdx.z];
							// get_Field_LAP(test_fp,x-4, y, z-4) = fp_shared[threadIdx.x+3 + 24*threadIdx.y + 96*threadIdx.z];
							// get_Field_LAP(test_fm,x-4, y, z-4) = fp_shared[threadIdx.x+4 + 24*threadIdx.y + 96*threadIdx.z];
							// get_Field_LAP(test_x, x-4, y, z-4) = fp_shared[threadIdx.x+5 + 24*threadIdx.y + 96*threadIdx.z];
							// get_Field_LAP(test_y, x-4, y, z-4) = fp_shared[threadIdx.x+6 + 24*threadIdx.y + 96*threadIdx.z];
							// get_Field_LAP(test_z, x-4, y, z-4) = fp_shared[threadIdx.x+7 + 24*threadIdx.y + 96*threadIdx.z];
					// **************************************************
				{
					// x_rhs1+
					__syncthreads();
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					if( x-4 == 0 ) 									tmp_l = 							    fp_shared[4+offset] ;
					if( x-4 == 1 ) 									tmp_l = OCFD_weno5_kernel_P_lift      (&fp_shared[1+offset]);
					if( x-4 == 2 ) 									tmp_l = OCFD_weno5_kernel_P_lift_plus (&fp_shared[1+offset]);
					if( x-4 == 3 || x-4 == job.end.x-job.start.x-2) tmp_l = OCFD_weno5_kernel_P			  (&fp_shared[1+offset]);
					if( x-4 >= 4 && x-4 <= job.end.x-job.start.x-3) tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);
					if(				x-4 == job.end.x-job.start.x-1) tmp_l = OCFD_weno5_kernel_P_right_plus(&fp_shared[1+offset]);
					if(				x-4 == job.end.x-job.start.x  ) tmp_l =   1.5*fp_shared[3+offset] - 0.5*fp_shared[2+offset] ;
					
					tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
					rhs1 += -Ajacobi*(tmp_r - tmp_l) *hx_d_rcp; // put_du_p_kernel
				}
				
				{
					// x_rhs1-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					if( x-4 == 0 )												tmp_l = OCFD_weno5_kernel_M_lift      (&fm_shared[2+offset]);
					if( x-4 == 1 )												tmp_l = OCFD_weno5_kernel_M_lift_plus (&fm_shared[2+offset]);
					if( x-4 == 2 || x-4 == 3 || x-4 == job.end.x-job.start.x-3) tmp_l = OCFD_weno5_kernel_M			  (&fm_shared[2+offset]);
					if( 			x-4 >= 4 && x-4 <= job.end.x-job.start.x-4) tmp_l = OCFD_weno7_SYMBO_kernel_M_opt (&fm_shared[  offset]);
					if(							x-4 == job.end.x-job.start.x-2)	tmp_l = OCFD_weno5_kernel_M_right_plus(&fm_shared[2+offset]);
					if(							x-4 == job.end.x-job.start.x-1) tmp_l = 0.5 *fm_shared[4+offset] + 0.5 *fm_shared[3+offset];
					if(							x-4 == job.end.x-job.start.x  )	tmp_l = 0.5 *fm_shared[3+offset] + 0.5 *fm_shared[2+offset];
					// if(							x-4 == job.end.x-job.start.x  )	tmp_l = fm_shared[3+offset];

					tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
					rhs1 += -Ajacobi*(tmp_r - tmp_l) *hx_d_rcp; // put_du_m_kernel
				}
			}
			
			// /*
			//---------------------
			// computing rhs2
			//---------------------
			for(int ii=0; ii<2; ii++)
			{
				//-------------		
				// x-dir
				//-------------
				REAL ss, Ak1;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0, uc1, uc2;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);
				Ak1 = stencil_Ax[ii] / ss;

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
				uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
				
				fp_shared[7*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_u[ii] + E2P * uc1 + E3P * uc2);
				fm_shared[7*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_u[ii] + E2M * uc1 + E3M * uc2);
			}
			
			{
				__syncthreads();
				
				{
					// x_rhs2+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					if( x-4 == 0 ) 									tmp_l = 							    fp_shared[4+offset] ;
					if( x-4 == 1 ) 									tmp_l = OCFD_weno5_kernel_P_lift      (&fp_shared[1+offset]);
					if( x-4 == 2 ) 									tmp_l = OCFD_weno5_kernel_P_lift_plus (&fp_shared[1+offset]);
					if( x-4 == 3 || x-4 == job.end.x-job.start.x-2) tmp_l = OCFD_weno5_kernel_P			  (&fp_shared[1+offset]);
					if( x-4 >= 4 && x-4 <= job.end.x-job.start.x-3) tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);
					if(				x-4 == job.end.x-job.start.x-1) tmp_l = OCFD_weno5_kernel_P_right_plus(&fp_shared[1+offset]);
					if(				x-4 == job.end.x-job.start.x  ) tmp_l =   1.5*fp_shared[3+offset] - 0.5*fp_shared[2+offset] ;
					
					tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
					rhs2 += -Ajacobi*(tmp_r - tmp_l) *hx_d_rcp; // put_du_p_kernel
				}
				
				{
					// x_rhs2-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					if( x-4 == 0 )												tmp_l = OCFD_weno5_kernel_M_lift      (&fm_shared[2+offset]);
					if( x-4 == 1 )												tmp_l = OCFD_weno5_kernel_M_lift_plus (&fm_shared[2+offset]);
					if( x-4 == 2 || x-4 == 3 || x-4 == job.end.x-job.start.x-3) tmp_l = OCFD_weno5_kernel_M			  (&fm_shared[2+offset]);
					if( 			x-4 >= 4 && x-4 <= job.end.x-job.start.x-4) tmp_l = OCFD_weno7_SYMBO_kernel_M_opt (&fm_shared[  offset]);
					if(							x-4 == job.end.x-job.start.x-2)	tmp_l = OCFD_weno5_kernel_M_right_plus(&fm_shared[2+offset]);
					if(							x-4 == job.end.x-job.start.x-1) tmp_l = 0.5 *fm_shared[4+offset] + 0.5 *fm_shared[3+offset];
					if(							x-4 == job.end.x-job.start.x  )	tmp_l = 0.5 *fm_shared[3+offset] + 0.5 *fm_shared[2+offset];
					// if(							x-4 == job.end.x-job.start.x  )	tmp_l = fm_shared[3+offset];

					tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
					rhs2 += -Ajacobi*(tmp_r - tmp_l) *hx_d_rcp; // put_du_m_kernel
				}
			}
			
			//---------------------
			// computing rhs3
			//---------------------
			for(int ii=0; ii<2; ii++)
			{
				//-------------		
				// x-dir
				//-------------
				REAL ss, Ak2;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0, vc1, vc2;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);
				Ak2 = stencil_Ay[ii] / ss;

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
				vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
				
				fp_shared[7*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_v[ii] + E2P * vc1 + E3P * vc2);
				fm_shared[7*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_v[ii] + E2M * vc1 + E3M * vc2);
			}
			
			{
				__syncthreads();
				{
					// x_rhs3+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					if( x-4 == 0 ) 									tmp_l = 							    fp_shared[4+offset] ;
					if( x-4 == 1 ) 									tmp_l = OCFD_weno5_kernel_P_lift      (&fp_shared[1+offset]);
					if( x-4 == 2 ) 									tmp_l = OCFD_weno5_kernel_P_lift_plus (&fp_shared[1+offset]);
					if( x-4 == 3 || x-4 == job.end.x-job.start.x-2) tmp_l = OCFD_weno5_kernel_P			  (&fp_shared[1+offset]);
					if( x-4 >= 4 && x-4 <= job.end.x-job.start.x-3) tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);
					if(				x-4 == job.end.x-job.start.x-1) tmp_l = OCFD_weno5_kernel_P_right_plus(&fp_shared[1+offset]);
					if(				x-4 == job.end.x-job.start.x  ) tmp_l =   1.5*fp_shared[3+offset] - 0.5*fp_shared[2+offset] ;
					
					tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
					rhs3 += -Ajacobi*(tmp_r - tmp_l) *hx_d_rcp; // put_du_p_kernel
				}
				
				{
					// x_rhs3-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					if( x-4 == 0 )												tmp_l = OCFD_weno5_kernel_M_lift      (&fm_shared[2+offset]);
					if( x-4 == 1 )												tmp_l = OCFD_weno5_kernel_M_lift_plus (&fm_shared[2+offset]);
					if( x-4 == 2 || x-4 == 3 || x-4 == job.end.x-job.start.x-3) tmp_l = OCFD_weno5_kernel_M			  (&fm_shared[2+offset]);
					if( 			x-4 >= 4 && x-4 <= job.end.x-job.start.x-4) tmp_l = OCFD_weno7_SYMBO_kernel_M_opt (&fm_shared[  offset]);
					if(							x-4 == job.end.x-job.start.x-2)	tmp_l = OCFD_weno5_kernel_M_right_plus(&fm_shared[2+offset]);
					if(							x-4 == job.end.x-job.start.x-1) tmp_l = 0.5 *fm_shared[4+offset] + 0.5 *fm_shared[3+offset];
					if(							x-4 == job.end.x-job.start.x  )	tmp_l = 0.5 *fm_shared[3+offset] + 0.5 *fm_shared[2+offset];
					// if(							x-4 == job.end.x-job.start.x  )	tmp_l = fm_shared[3+offset];

					tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
					rhs3 += -Ajacobi*(tmp_r - tmp_l) *hx_d_rcp; // put_du_m_kernel
				}
			}
			
			//---------------------
			// computing rhs4
			//---------------------
			for(int ii=0; ii<2; ii++)
			{
				//-------------		
				// x-dir
				//-------------
				REAL ss, Ak3;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0, wc1, wc2;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);
				Ak3 = stencil_Az[ii] / ss;

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
				wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3;
				
				fp_shared[7*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_w[ii] + E2P * wc1 + E3P * wc2);
				fm_shared[7*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_w[ii] + E2M * wc1 + E3M * wc2);
			}
			
			{
				__syncthreads();
				{
					// x_rhs4+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					if( x-4 == 0 ) 									tmp_l = 							    fp_shared[4+offset] ;
					if( x-4 == 1 ) 									tmp_l = OCFD_weno5_kernel_P_lift      (&fp_shared[1+offset]);
					if( x-4 == 2 ) 									tmp_l = OCFD_weno5_kernel_P_lift_plus (&fp_shared[1+offset]);
					if( x-4 == 3 || x-4 == job.end.x-job.start.x-2) tmp_l = OCFD_weno5_kernel_P			  (&fp_shared[1+offset]);
					if( x-4 >= 4 && x-4 <= job.end.x-job.start.x-3) tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);
					if(				x-4 == job.end.x-job.start.x-1) tmp_l = OCFD_weno5_kernel_P_right_plus(&fp_shared[1+offset]);
					if(				x-4 == job.end.x-job.start.x  ) tmp_l =   1.5*fp_shared[3+offset] - 0.5*fp_shared[2+offset] ;
					
					tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
					rhs4 += -Ajacobi*(tmp_r - tmp_l) *hx_d_rcp; // put_du_p_kernel
				}
				
				{
					// x_rhs4-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					if( x-4 == 0 )												tmp_l = OCFD_weno5_kernel_M_lift      (&fm_shared[2+offset]);
					if( x-4 == 1 )												tmp_l = OCFD_weno5_kernel_M_lift_plus (&fm_shared[2+offset]);
					if( x-4 == 2 || x-4 == 3 || x-4 == job.end.x-job.start.x-3) tmp_l = OCFD_weno5_kernel_M			  (&fm_shared[2+offset]);
					if( 			x-4 >= 4 && x-4 <= job.end.x-job.start.x-4) tmp_l = OCFD_weno7_SYMBO_kernel_M_opt (&fm_shared[  offset]);
					if(							x-4 == job.end.x-job.start.x-2)	tmp_l = OCFD_weno5_kernel_M_right_plus(&fm_shared[2+offset]);
					if(							x-4 == job.end.x-job.start.x-1) tmp_l = 0.5 *fm_shared[4+offset] + 0.5 *fm_shared[3+offset];
					if(							x-4 == job.end.x-job.start.x  )	tmp_l = 0.5 *fm_shared[3+offset] + 0.5 *fm_shared[2+offset];
					// if(							x-4 == job.end.x-job.start.x  )	tmp_l = fm_shared[3+offset];

					tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
					rhs4 += -Ajacobi*(tmp_r - tmp_l) *hx_d_rcp; // put_du_m_kernel
					// if( x-4 == job.end.x-job.start.x-1 ) rhs4 += -Ajacobi*(tmp_r - tmp_l) *hx_d_rcp; // put_du_m_kernel
				}
			}
			
			//---------------------
			// computing rhs5
			//---------------------
			for(int ii=0; ii<2; ii++)
			{
				//-------------		
				// x-dir
				//-------------
				REAL ss, Ak1, Ak2, Ak3;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);
				Ak1 = stencil_Ax[ii] / ss;
				Ak2 = stencil_Ay[ii] / ss;
				Ak3 = stencil_Az[ii] / ss;

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
				uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
				vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
				vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
				wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
				wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3;
				vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
				vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50;
				vv = (Gamma_d - 1.0) * (stencil_u[ii] * stencil_u[ii] 
									  + stencil_v[ii] * stencil_v[ii]
									  + stencil_w[ii] * stencil_w[ii] );
				W2 = split_C3_d * stencil_cc[ii] * stencil_cc[ii];
				
				fp_shared[7*ii +offset] = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P));
				fm_shared[7*ii +offset] = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
			}
			
			{
				__syncthreads();
				{
					// x_rhs5+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					if( x-4 == 0 ) 									tmp_l = 							    fp_shared[4+offset] ;
					if( x-4 == 1 ) 									tmp_l = OCFD_weno5_kernel_P_lift      (&fp_shared[1+offset]);
					if( x-4 == 2 ) 									tmp_l = OCFD_weno5_kernel_P_lift_plus (&fp_shared[1+offset]);
					if( x-4 == 3 || x-4 == job.end.x-job.start.x-2) tmp_l = OCFD_weno5_kernel_P			  (&fp_shared[1+offset]);
					if( x-4 >= 4 && x-4 <= job.end.x-job.start.x-3) tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);
					if(				x-4 == job.end.x-job.start.x-1) tmp_l = OCFD_weno5_kernel_P_right_plus(&fp_shared[1+offset]);
					if(				x-4 == job.end.x-job.start.x  ) tmp_l =   1.5*fp_shared[3+offset] - 0.5*fp_shared[2+offset] ;
					
					tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
					rhs5 += -Ajacobi*(tmp_r - tmp_l) *hx_d_rcp; // put_du_p_kernel
				}
				
				{
					// x_rhs5-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					if( x-4 == 0 )												tmp_l = OCFD_weno5_kernel_M_lift      (&fm_shared[2+offset]);
					if( x-4 == 1 )												tmp_l = OCFD_weno5_kernel_M_lift_plus (&fm_shared[2+offset]);
					if( x-4 == 2 || x-4 == 3 || x-4 == job.end.x-job.start.x-3) tmp_l = OCFD_weno5_kernel_M			  (&fm_shared[2+offset]);
					if( 			x-4 >= 4 && x-4 <= job.end.x-job.start.x-4) tmp_l = OCFD_weno7_SYMBO_kernel_M_opt (&fm_shared[  offset]);
					if(							x-4 == job.end.x-job.start.x-2)	tmp_l = OCFD_weno5_kernel_M_right_plus(&fm_shared[2+offset]);
					if(							x-4 == job.end.x-job.start.x-1) tmp_l = 0.5 *fm_shared[4+offset] + 0.5 *fm_shared[3+offset];
					if(							x-4 == job.end.x-job.start.x  )	tmp_l = 0.5 *fm_shared[3+offset] + 0.5 *fm_shared[2+offset];
					// if(							x-4 == job.end.x-job.start.x  )	tmp_l = fm_shared[3+offset];

					tmp_r = __shfl_down_double(tmp_l, 1, hipWarpSize);
					rhs5 += -Ajacobi*(tmp_r - tmp_l) *hx_d_rcp; // put_du_m_kernel
				}
			}
		}
	}

	// /*
	//
	//======================================
	// 			for Y-direction
	//======================================
	{
		{
			// int blk = blockIdx.y / (gridDim.y-1); // the last block.
			// int thd = threadIdx.y / ( job.end.y-job.start.y-(blockDim.y-1)*(gridDim.y-1) ); // the threads which overstep the boundary.
			// int flag = blk*thd; // data prepare without the "if".
			// int flag = threadIdx.y / (blockDim.y-1); // int flag = y / job.end.y;
			
			//
			//for rho parameters
			stencil_d[0] = get_Field_LAP(d, x, y-4, z);
			stencil_d[1] = d_shared[1+threadIdx.x+16*threadIdx.y+64*threadIdx.z];
			stencil_d[2] = get_Field_LAP(d, x, y+4, z);
			// stencil_d[2] = get_Field_LAP(d, x, y+4-flag*8, z);

			//
			//for u parameters
			stencil_u[0] = get_Field_LAP(u, x, y-4, z);
			stencil_u[1] = u_shared[1+threadIdx.x+16*threadIdx.y+64*threadIdx.z];
			stencil_u[2] = get_Field_LAP(u, x, y+4, z);
			
			//
			//for v parameters
			stencil_v[0] = get_Field_LAP(v, x, y-4, z);
			stencil_v[1] = v_shared[1+threadIdx.x+16*threadIdx.y+64*threadIdx.z];
			stencil_v[2] = get_Field_LAP(v, x, y+4, z);
			
			//
			//for w parameters
			stencil_w[0] = get_Field_LAP(w, x, y-4, z);
			stencil_w[1] = w_shared[1+threadIdx.x+16*threadIdx.y+64*threadIdx.z];
			stencil_w[2] = get_Field_LAP(w, x, y+4, z);
			
			//
			//for T/cc parameters
			stencil_cc[0] = get_Field_LAP(cc, x, y-4, z);
			stencil_cc[1] = cc_shared[1+threadIdx.x+16*threadIdx.y+64*threadIdx.z];
			stencil_cc[2] = get_Field_LAP(cc, x, y+4, z);
			
			//
			//for Ax parameters
			stencil_Ax[0] = get_Field_LAP(Bx, x, y-4, z);
			stencil_Ax[1] = get_Field_LAP(Bx, x, y,   z);
			stencil_Ax[2] = get_Field_LAP(Bx, x, y+4, z);
			
			//	
			//for Ay parameters
			stencil_Ay[0] = get_Field_LAP(By, x, y-4, z);
			stencil_Ay[1] = get_Field_LAP(By, x, y,   z);
			stencil_Ay[2] = get_Field_LAP(By, x, y+4, z);
			
			//		
			//for Az parameters
			stencil_Az[0] = get_Field_LAP(Bz, x, y-4, z);
			stencil_Az[1] = get_Field_LAP(Bz, x, y,   z);
			stencil_Az[2] = get_Field_LAP(Bz, x, y+4, z);
		}
		
		// if( x>=job.start.x && x<=job.end.x && y>=job.start.y && y<=job.end.y && z>=job.start.z && z<=job.end.z)
		// if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-5)
		{
			int offset = threadIdx.y+12*threadIdx.z+48*threadIdx.x; // take y as the continuous direction in LDS. y-z-x: 12*4*16.
			// int flag = (threadIdx.y+blockDim.y-1)/blockDim.y; // threadIdx.y = 0, flag = 0; otherwise flag = 1.
			
			//---------------------
			// computing rhs1
			//---------------------
			for(int ii=0; ii<3; ii++)
			{
				//-------------		
				// y-dir
				//-------------
				REAL ss;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				
				// take z as the continuous direction in LDS. z-y-x: 12*4*16
				fp_shared[4*ii +offset] = tmp0 * (split_C1_d * E1P + E2P + E3P);
				fm_shared[4*ii +offset] = tmp0 * (split_C1_d * E1M + E2M + E3M);
			}
			
			{
				// REAL rhs = rhs1_shared[threadIdx.x+16*threadIdx.y+64*threadIdx.z];
				__syncthreads();
				
				{	
					// y_rhs1+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					if( y-4 == 0 )									{tmp_r = OCFD_weno5_kernel_P_lift      (&fp_shared[2+offset]);
																	 tmp_l = 							     fp_shared[4+offset] ;}
																	 
					if( y-4 == 1 )									{tmp_r = OCFD_weno5_kernel_P_lift_plus (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_lift      (&fp_shared[1+offset]);}
																	 
					if( y-4 == 2 )									{tmp_r = OCFD_weno5_kernel_P		   (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_lift_plus (&fp_shared[1+offset]);}
																	 
					if( y-4 == 3 )									{tmp_r = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[1+offset]);
																	 tmp_l = OCFD_weno5_kernel_P		   (&fp_shared[1+offset]);}
																	 
					if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-4) {tmp_r = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[1+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);}
																	 
					if(				y-4 == job.end.y-job.start.y-3) {tmp_r = OCFD_weno5_kernel_P		   (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);}
																	 
																	 
					if(				y-4 == job.end.y-job.start.y-2) {tmp_r = OCFD_weno5_kernel_P_right_plus(&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P		   (&fp_shared[1+offset]);}
					
					if(				y-4 == job.end.y-job.start.y-1) {tmp_r = fp_shared[4+offset]+0.5*minmod2(fp_shared[4+offset] -fp_shared[3+offset], 
																											 fp_shared[4+offset] -fp_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_right_plus(&fp_shared[1+offset]);}
					
					rhs1 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; // put_du_p_kernel
				}
				
				{
					// y_rhs1-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					if( y-4 == 0 )									{tmp_r = OCFD_weno5_kernel_M_lift_plus	(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_lift		(&fm_shared[2+offset]);
																	 rhs1 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 1 )									{tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_lift_plus	(&fm_shared[2+offset]);
																	 rhs1 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 2 ) 									{tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs1 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 3 )									{tmp_r = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[1+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs1 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-5) {tmp_r = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[1+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[  offset]);
																	 rhs1 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-4) {tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[  offset]);
																	 rhs1 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-3) {tmp_r = OCFD_weno5_kernel_M_right_plus (&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs1 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-2) {tmp_r = fm_shared[5+offset]-0.5*minmod2(fm_shared[5+offset] - fm_shared[4+offset], 
																											 fm_shared[5+offset] - fm_shared[4+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_right_plus(&fm_shared[2+offset]);
																	 rhs1 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-1) {tmp_r =								 fm_shared[4+offset];
																	 tmp_l = fm_shared[4+offset]-0.5*minmod2(fm_shared[4+offset] - fm_shared[3+offset], 
																											 fm_shared[4+offset] - fm_shared[3+offset]);}
				}
			}
			
			//---------------------
			// computing rhs2
			//---------------------
			// __syncthreads();
			for(int ii=0; ii<3; ii++)
			{
				//-------------		
				// y-dir
				//-------------
				REAL ss, Ak1;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0, uc1, uc2;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);
				Ak1 = stencil_Ax[ii] / ss;

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
				uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
				
				fp_shared[4*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_u[ii] + E2P * uc1 + E3P * uc2);
				fm_shared[4*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_u[ii] + E2M * uc1 + E3M * uc2);
			}
			
			{
				// REAL rhs = rhs2_shared[threadIdx.x+16*threadIdx.y+64*threadIdx.z];
				__syncthreads();

				{	
					// y_rhs2+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					if( y-4 == 0 )									{tmp_r = OCFD_weno5_kernel_P_lift      (&fp_shared[2+offset]);
																	 tmp_l = 							     fp_shared[4+offset] ;}
																	 
					if( y-4 == 1 )									{tmp_r = OCFD_weno5_kernel_P_lift_plus (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_lift      (&fp_shared[1+offset]);}
																	 
					if( y-4 == 2 )									{tmp_r = OCFD_weno5_kernel_P		   (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_lift_plus (&fp_shared[1+offset]);}
																	 
					if( y-4 == 3 )									{tmp_r = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[1+offset]);
																	 tmp_l = OCFD_weno5_kernel_P		   (&fp_shared[1+offset]);}
																	 
					if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-4) {tmp_r = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[1+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);}
																	 
					if(				y-4 == job.end.y-job.start.y-3) {tmp_r = OCFD_weno5_kernel_P		   (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);}
																	 
																	 
					if(				y-4 == job.end.y-job.start.y-2) {tmp_r = OCFD_weno5_kernel_P_right_plus(&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P		   (&fp_shared[1+offset]);}
					
					if(				y-4 == job.end.y-job.start.y-1) {tmp_r = fp_shared[4+offset]+0.5*minmod2(fp_shared[4+offset] -fp_shared[3+offset], 
																											 fp_shared[4+offset] -fp_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_right_plus(&fp_shared[1+offset]);}
					
					rhs2 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; // put_du_p_kernel
				}
				
				{
					// y_rhs2-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					if( y-4 == 0 )									{tmp_r = OCFD_weno5_kernel_M_lift_plus	(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_lift		(&fm_shared[2+offset]);
																	 rhs2 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 1 )									{tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_lift_plus	(&fm_shared[2+offset]);
																	 rhs2 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 2 ) 									{tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs2 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 3 )									{tmp_r = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[1+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs2 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-5) {tmp_r = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[1+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[  offset]);
																	 rhs2 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-4) {tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[  offset]);
																	 rhs2 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-3) {tmp_r = OCFD_weno5_kernel_M_right_plus (&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs2 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-2) {tmp_r = fm_shared[5+offset]-0.5*minmod2(fm_shared[5+offset] - fm_shared[4+offset], 
																											 fm_shared[5+offset] - fm_shared[4+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_right_plus(&fm_shared[2+offset]);
																	 rhs2 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-1) {tmp_r =								 fm_shared[4+offset];
																	 tmp_l = fm_shared[4+offset]-0.5*minmod2(fm_shared[4+offset] - fm_shared[3+offset], 
																											 fm_shared[4+offset] - fm_shared[3+offset]);}
				}
			}
			
			//---------------------
			// computing rhs3
			//---------------------
			// __syncthreads();
			for(int ii=0; ii<3; ii++)
			{
				//-------------		
				// y-dir
				//-------------
				REAL ss, Ak2;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0, vc1, vc2;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);
				Ak2 = stencil_Ay[ii] / ss;

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
				vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
				
				fp_shared[4*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_v[ii] + E2P * vc1 + E3P * vc2);
				fm_shared[4*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_v[ii] + E2M * vc1 + E3M * vc2);
			}
			
			{
				// REAL rhs = rhs3_shared[threadIdx.x+16*threadIdx.y+64*threadIdx.z];
				__syncthreads();

				{	
					// y_rhs3+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					if( y-4 == 0 )									{tmp_r = OCFD_weno5_kernel_P_lift      (&fp_shared[2+offset]);
																	 tmp_l = 							     fp_shared[4+offset] ;}
																	 
					if( y-4 == 1 )									{tmp_r = OCFD_weno5_kernel_P_lift_plus (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_lift      (&fp_shared[1+offset]);}
																	 
					if( y-4 == 2 )									{tmp_r = OCFD_weno5_kernel_P		   (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_lift_plus (&fp_shared[1+offset]);}
																	 
					if( y-4 == 3 )									{tmp_r = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[1+offset]);
																	 tmp_l = OCFD_weno5_kernel_P		   (&fp_shared[1+offset]);}
																	 
					if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-4) {tmp_r = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[1+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);}
																	 
					if(				y-4 == job.end.y-job.start.y-3) {tmp_r = OCFD_weno5_kernel_P		   (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);}
																	 
																	 
					if(				y-4 == job.end.y-job.start.y-2) {tmp_r = OCFD_weno5_kernel_P_right_plus(&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P		   (&fp_shared[1+offset]);}
					
					if(				y-4 == job.end.y-job.start.y-1) {tmp_r = fp_shared[4+offset]+0.5*minmod2(fp_shared[4+offset] -fp_shared[3+offset], 
																											 fp_shared[4+offset] -fp_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_right_plus(&fp_shared[1+offset]);}
					
					rhs3 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; // put_du_p_kernel
				}
				
				{
					// y_rhs3-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					if( y-4 == 0 )									{tmp_r = OCFD_weno5_kernel_M_lift_plus	(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_lift		(&fm_shared[2+offset]);
																	 rhs3 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 1 )									{tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_lift_plus	(&fm_shared[2+offset]);
																	 rhs3 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 2 ) 									{tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs3 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 3 )									{tmp_r = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[1+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs3 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-5) {tmp_r = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[1+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[  offset]);
																	 rhs3 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-4) {tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[  offset]);
																	 rhs3 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-3) {tmp_r = OCFD_weno5_kernel_M_right_plus (&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs3 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-2) {tmp_r = fm_shared[5+offset]-0.5*minmod2(fm_shared[5+offset] - fm_shared[4+offset], 
																											 fm_shared[5+offset] - fm_shared[4+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_right_plus(&fm_shared[2+offset]);
																	 rhs3 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-1) {tmp_r =								 fm_shared[4+offset];
																	 tmp_l = fm_shared[4+offset]-0.5*minmod2(fm_shared[4+offset] - fm_shared[3+offset], 
																											 fm_shared[4+offset] - fm_shared[3+offset]);}
				}
			}
			
			//---------------------
			// computing rhs4
			//---------------------
			// __syncthreads();
			for(int ii=0; ii<3; ii++)
			{
				//-------------		
				// y-dir
				//-------------
				REAL ss, Ak3;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0, wc1, wc2;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);
				Ak3 = stencil_Az[ii] / ss;

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
				wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3;
				
				fp_shared[4*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_w[ii] + E2P * wc1 + E3P * wc2);
				fm_shared[4*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_w[ii] + E2M * wc1 + E3M * wc2);
			}
			
			{
				// REAL rhs = rhs4_shared[threadIdx.x+16*threadIdx.y+64*threadIdx.z];
				__syncthreads();

				{	
					// y_rhs4+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					if( y-4 == 0 )									{tmp_r = OCFD_weno5_kernel_P_lift      (&fp_shared[2+offset]);
																	 tmp_l = 							     fp_shared[4+offset] ;}
																	 
					if( y-4 == 1 )									{tmp_r = OCFD_weno5_kernel_P_lift_plus (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_lift      (&fp_shared[1+offset]);}
																	 
					if( y-4 == 2 )									{tmp_r = OCFD_weno5_kernel_P		   (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_lift_plus (&fp_shared[1+offset]);}
																	 
					if( y-4 == 3 )									{tmp_r = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[1+offset]);
																	 tmp_l = OCFD_weno5_kernel_P		   (&fp_shared[1+offset]);}
																	 
					if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-4) {tmp_r = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[1+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);}
																	 
					if(				y-4 == job.end.y-job.start.y-3) {tmp_r = OCFD_weno5_kernel_P		   (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);}
																	 
																	 
					if(				y-4 == job.end.y-job.start.y-2) {tmp_r = OCFD_weno5_kernel_P_right_plus(&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P		   (&fp_shared[1+offset]);}
					
					if(				y-4 == job.end.y-job.start.y-1) {tmp_r = fp_shared[4+offset]+0.5*minmod2(fp_shared[4+offset] -fp_shared[3+offset], 
																											 fp_shared[4+offset] -fp_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_right_plus(&fp_shared[1+offset]);}
					
					rhs4 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; // put_du_p_kernel
				}
				
				{
					// y_rhs4-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					if( y-4 == 0 )									{tmp_r = OCFD_weno5_kernel_M_lift_plus	(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_lift		(&fm_shared[2+offset]);
																	 rhs4 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 1 )									{tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_lift_plus	(&fm_shared[2+offset]);
																	 rhs4 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 2 ) 									{tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs4 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 3 )									{tmp_r = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[1+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs4 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-5) {tmp_r = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[1+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[  offset]);
																	 rhs4 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-4) {tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[  offset]);
																	 rhs4 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-3) {tmp_r = OCFD_weno5_kernel_M_right_plus (&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs4 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-2) {tmp_r = fm_shared[5+offset]-0.5*minmod2(fm_shared[5+offset] - fm_shared[4+offset], 
																											 fm_shared[5+offset] - fm_shared[4+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_right_plus(&fm_shared[2+offset]);
																	 rhs4 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-1) {tmp_r =								 fm_shared[4+offset];
																	 tmp_l = fm_shared[4+offset]-0.5*minmod2(fm_shared[4+offset] - fm_shared[3+offset], 
																											 fm_shared[4+offset] - fm_shared[3+offset]);}
				}
			}
			
			//---------------------
			// computing rhs5
			//---------------------
			// __syncthreads();
			for(int ii=0; ii<3; ii++)
			{
				//-------------		
				// y-dir
				//-------------
				REAL ss, Ak1, Ak2, Ak3;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);
				Ak1 = stencil_Ax[ii] / ss;
				Ak2 = stencil_Ay[ii] / ss;
				Ak3 = stencil_Az[ii] / ss;

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
				uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
				vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
				vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
				wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
				wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3;
				vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
				vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50;
				vv = (Gamma_d - 1.0) * (stencil_u[ii] * stencil_u[ii] 
									  + stencil_v[ii] * stencil_v[ii]
									  + stencil_w[ii] * stencil_w[ii] );
				W2 = split_C3_d * stencil_cc[ii] * stencil_cc[ii];
				
				fp_shared[4*ii +offset] = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P));
				fm_shared[4*ii +offset] = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
			}
			
			{
				// REAL rhs = rhs5_shared[threadIdx.x+16*threadIdx.y+64*threadIdx.z];
				__syncthreads();
				
				{	
					// y_rhs5+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					if( y-4 == 0 )									{tmp_r = OCFD_weno5_kernel_P_lift      (&fp_shared[2+offset]);
																	 tmp_l = 							     fp_shared[4+offset] ;}
																	 
					if( y-4 == 1 )									{tmp_r = OCFD_weno5_kernel_P_lift_plus (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_lift      (&fp_shared[1+offset]);}
																	 
					if( y-4 == 2 )									{tmp_r = OCFD_weno5_kernel_P		   (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_lift_plus (&fp_shared[1+offset]);}
																	 
					if( y-4 == 3 )									{tmp_r = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[1+offset]);
																	 tmp_l = OCFD_weno5_kernel_P		   (&fp_shared[1+offset]);}
																	 
					if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-4) {tmp_r = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[1+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);}
																	 
					if(				y-4 == job.end.y-job.start.y-3) {tmp_r = OCFD_weno5_kernel_P		   (&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_P_opt (&fp_shared[  offset]);}
																	 
																	 
					if(				y-4 == job.end.y-job.start.y-2) {tmp_r = OCFD_weno5_kernel_P_right_plus(&fp_shared[2+offset]);
																	 tmp_l = OCFD_weno5_kernel_P		   (&fp_shared[1+offset]);}
					
					if(				y-4 == job.end.y-job.start.y-1) {tmp_r = fp_shared[4+offset]+0.5*minmod2(fp_shared[4+offset] -fp_shared[3+offset], 
																											 fp_shared[4+offset] -fp_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_P_right_plus(&fp_shared[1+offset]);}
					
					rhs5 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; // put_du_p_kernel
				}
				
				{
					// y_rhs5-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					if( y-4 == 0 )									{tmp_r = OCFD_weno5_kernel_M_lift_plus	(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_lift		(&fm_shared[2+offset]);
																	 rhs5 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 1 )									{tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_lift_plus	(&fm_shared[2+offset]);
																	 rhs5 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 2 ) 									{tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs5 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 == 3 )									{tmp_r = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[1+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs5 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if( y-4 >= 4 && y-4 <= job.end.y-job.start.y-5) {tmp_r = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[1+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[  offset]);
																	 rhs5 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-4) {tmp_r = OCFD_weno5_kernel_M			(&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno7_SYMBO_kernel_M_opt	(&fm_shared[  offset]);
																	 rhs5 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-3) {tmp_r = OCFD_weno5_kernel_M_right_plus (&fm_shared[3+offset]);
																	 tmp_l = OCFD_weno5_kernel_M			(&fm_shared[2+offset]);
																	 rhs5 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-2) {tmp_r = fm_shared[5+offset]-0.5*minmod2(fm_shared[5+offset] - fm_shared[4+offset], 
																											 fm_shared[5+offset] - fm_shared[4+offset]);
																	 tmp_l = OCFD_weno5_kernel_M_right_plus(&fm_shared[2+offset]);
																	 rhs5 += -Ajacobi*(tmp_r - tmp_l)*hy_d_rcp; }
					
					if(				y-4 == job.end.y-job.start.y-1) {tmp_r =								 fm_shared[4+offset];
																	 tmp_l = fm_shared[4+offset]-0.5*minmod2(fm_shared[4+offset] - fm_shared[3+offset], 
																											 fm_shared[4+offset] - fm_shared[3+offset]);}
				}
			}	
		}
		
	}
	// */
	
	// /*
	//======================================
	// 			for Z-direction
	//======================================
	{
		{
			// int blk = blockIdx.z / (gridDim.z-1); // the last block.
			// int thd = threadIdx.z / ( job.end.z-job.start.z-(blockDim.z-1)*(gridDim.z-1) ); // the threads which overstep the boundary.
			// int flag = blk*thd; // data prepare without the "if".
			// int flag = threadIdx.z / (blockDim.z-1); // int flag = z / job.end.z;
			
			//
			//for rho parameters
			// __syncthreads();
			stencil_d[0] = get_Field_LAP(d, x, y, z-4);
			stencil_d[2] = get_Field_LAP(d, x, y, z+4);
			// stencil_d[2] = get_Field_LAP(d, x, y, z+4-flag*8);
			
			//
			//for u parameters
			stencil_u[0] = get_Field_LAP(u, x, y, z-4) ;
			stencil_u[2] = get_Field_LAP(u, x, y, z+4);
			
			//
			//for v parameters
			stencil_v[0] = get_Field_LAP(v, x, y, z-4) ;
			stencil_v[2] = get_Field_LAP(v, x, y, z+4);
		
			//
			//for w parameters
			stencil_w[0] = get_Field_LAP(w, x, y, z-4) ;
			stencil_w[2] = get_Field_LAP(w, x, y, z+4);
			
			//
			//for T/cc parameters
			stencil_cc[0] = get_Field_LAP(cc, x, y, z-4);
			stencil_cc[2] = get_Field_LAP(cc, x, y, z+4);
			
			//
			//for Ax parameters
			stencil_Ax[0] = get_Field_LAP(Cx, x, y, z-4) ;
			stencil_Ax[1] = get_Field_LAP(Cx, x, y, z  ) ;
			stencil_Ax[2] = get_Field_LAP(Cx, x, y, z+4);
			
			//
			//for Ay parameters
			stencil_Ay[0] = get_Field_LAP(Cy, x, y, z-4) ;
			stencil_Ay[1] = get_Field_LAP(Cy, x, y, z  ) ;
			stencil_Ay[2] = get_Field_LAP(Cy, x, y, z+4);
			
			//
			//for Az parameters
			stencil_Az[0] = get_Field_LAP(Cz, x, y, z-4) ;
			stencil_Az[1] = get_Field_LAP(Cz, x, y, z  ) ;
			stencil_Az[2] = get_Field_LAP(Cz, x, y, z+4);
			
		}
		
		// **************************************************
		
			// get_Field_LAP(test_x, x-4, y, z-4) = stencil_d[4];
			// get_Field_LAP(test_y, x-4, y, z-4) = stencil_u[4];
			// get_Field_LAP(test_z, x-4, y, z-4) = stencil_v[4];
			// get_Field_LAP(test_fp,x-4, y, z-4) = stencil_w[4];
			// get_Field_LAP(test_fm,x-4, y, z-4) = stencil_cc[4];
			
			// get_Field_LAP(test_x, x-4, y, z-4) = stencil_Ax[4];
			// get_Field_LAP(test_y, x-4, y, z-4) = stencil_Ay[4];
			// get_Field_LAP(test_z, x-4, y, z-4) = stencil_Az[4];
			// get_Field_LAP(test_fp,x-4, y, z-4) = ajac_shared[threadIdx.x+16*threadIdx.y+64*threadIdx.z];
			
			// get_Field_LAP(test_x, x-4, y, z-4) = (double)x;
			// get_Field_LAP(test_y, x-4, y, z-4) = (double)y;
			// get_Field_LAP(test_z, x-4, y, z-4) = (double)z;
			
			// get_Field_LAP(test_fp,x-4, y, z-4) =  (1000000*job.start.x+1000*job.start.y+job.start.z)*1.0 ;
			// get_Field_LAP(test_fm,x-4, y, z-4) =  (1000000*job.end.x+1000*job.end.y+job.end.z)*1.0 ;
		
		// **************************************************		
		
		//
		//computing the flux+ and flux -
		//store in stencil_fm and stencil_fp 
		
		// if( x>0 )
		// if( x>=job.start.x && x<=job.end.x && y>=job.start.y && y<=job.end.y && z>=job.start.z && z<=job.end.z)
		{
			int offset = threadIdx.z+12*threadIdx.y+48*threadIdx.x; // take z as the continuous direction in LDS. z-y-x: 12*4*16.
			int flag1 = (blockDim.x+14-threadIdx.x) / blockDim.x; // threadIdx.x > 14, flag = 0; otherwise flag = 1.
				// d_shared[ ( 1 +threadIdx.x +15*threadIdx.y +64*threadIdx.z) *flag1 ] = rhs1; // take x as the continuous direction in LDS. x-y-z: 15*4*4.
			int flag2 = threadIdx.x / (blockDim.x-1); // threadIdx.x = blockDim.x-1, flag = 1; otherwise flag = 0.
				// get_SoA(du, x-job.start.x-flag2, y-job.start.y, z-job.start.z, 0) = d_shared[ 1 +threadIdx.x +15*threadIdx.y +64*threadIdx.z - flag2 ];
				
			
			// int flag1 = 1.99-threadIdx.y/(blockDim.y-1); // threadIdx.y = blockDim.y-1, flag1 = 0; otherwise flag1 = 1.
			// int flag = (threadIdx.z+blockDim.z-1)/blockDim.z; // threadIdx.z = 0, flag = 0; otherwise flag = 1.
			// Ajacobi = ajac_shared[threadIdx.x +16*threadIdx.y +64*threadIdx.z];
			
			//---------------------
			// computing rhs1
			//---------------------
			for(int ii=0; ii<3; ii++)
			{
				//-------------		
				// z-dir
				//-------------
				REAL ss;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				
				// take z as the continuous direction in LDS. z-y-x: 12*4*16
				fp_shared[4*ii +offset] = tmp0 * (split_C1_d * E1P + E2P + E3P);
				fm_shared[4*ii +offset] = tmp0 * (split_C1_d * E1M + E2M + E3M);
			}
				
			{
				// REAL rhs = rhs1_shared[threadIdx.x +16*(threadIdx.y+1) +64*(threadIdx.z-1)*flag];
				// REAL rhs = rhs1_shared[threadIdx.x +16*(threadIdx.y+1) +64*threadIdx.z]; // without transferring data between threads.
				__syncthreads();
				
							// **************************************************
								// get_Field_LAP(test_x, x-4, y, z-4) = fm_shared[  offset];
								// get_Field_LAP(test_y, x-4, y, z-4) = fm_shared[1+offset];
								// get_Field_LAP(test_z, x-4, y, z-4) = fm_shared[2+offset];
								// get_Field_LAP(test_fp,x-4, y, z-4) = fm_shared[3+offset];
								// get_Field_LAP(test_fm,x-4, y, z-4) = fm_shared[4+offset];
								
								// get_Field_LAP(test_x, x-4, y, z-4) = fm_shared[5+offset];
								// get_Field_LAP(test_y, x-4, y, z-4) = fm_shared[6+offset];
								// get_Field_LAP(test_z, x-4, y, z-4) = fm_shared[7+offset];
							// **************************************************
				{	
					// z_rhs1+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					tmp_r = OCFD_weno7_SYMBO_kernel_P_opt(&fp_shared[offset+1]); // without transferring data between threads.
					tmp_l = OCFD_weno7_SYMBO_kernel_P_opt(&fp_shared[offset]);
					rhs1 += -Ajacobi*(tmp_r - tmp_l)*hz_d_rcp; // put_du_p_kernel
				}
				
				{
					// z_rhs1-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					tmp_r = OCFD_weno7_SYMBO_kernel_M_opt(&fm_shared[offset+1]); // without transferring data between threads.
					tmp_l = OCFD_weno7_SYMBO_kernel_M_opt(&fm_shared[offset]);
					rhs1 += -Ajacobi*(tmp_r - tmp_l)*hz_d_rcp; // put_du_m_kernel
					d_shared[ ( 1 +threadIdx.x +15*threadIdx.y +64*threadIdx.z) *flag1 ] = rhs1;
					// if(threadIdx.x != (blockDim.x-1)) get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = rhs1;
				}
			}
			
			//---------------------
			// computing rhs2
			//---------------------
			for(int ii=0; ii<3; ii++)
			{
				//-------------		
				// z-dir
				//-------------
				REAL ss, Ak1;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0, uc1, uc2;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);
				Ak1 = stencil_Ax[ii] / ss;

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
				uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
				
				fp_shared[4*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_u[ii] + E2P * uc1 + E3P * uc2);
				fm_shared[4*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_u[ii] + E2M * uc1 + E3M * uc2);
			}
			{
				// REAL rhs = rhs2_shared[threadIdx.x +16*(threadIdx.y+1) +64*(threadIdx.z-1)*flag];
				// REAL rhs = rhs2_shared[threadIdx.x +16*(threadIdx.y+1) +64*threadIdx.z];
				__syncthreads();
				
				{	
					// z_rhs2+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					tmp_r = OCFD_weno7_SYMBO_kernel_P_opt(&fp_shared[offset+1]); // without transferring data between threads.
					tmp_l = OCFD_weno7_SYMBO_kernel_P_opt(&fp_shared[offset]);
					rhs2 += -Ajacobi*(tmp_r - tmp_l)*hz_d_rcp; // put_du_p_kernel
					get_SoA(du, x-job.start.x-flag2, y-job.start.y, z-job.start.z, 0) = d_shared[ 1 +threadIdx.x +15*threadIdx.y +64*threadIdx.z - flag2 ]; // rhs1
				}
				
				{
					// z_rhs2-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					tmp_r = OCFD_weno7_SYMBO_kernel_M_opt(&fm_shared[offset+1]); // without transferring data between threads.
					tmp_l = OCFD_weno7_SYMBO_kernel_M_opt(&fm_shared[offset]);
					rhs2 += -Ajacobi*(tmp_r - tmp_l)*hz_d_rcp; // put_du_m_kernel
					u_shared[ ( 1 +threadIdx.x +15*threadIdx.y +64*threadIdx.z) *flag1 ] = rhs2;
					// if(threadIdx.x != (blockDim.x-1)) get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = rhs2;
				}
			}
			
			//---------------------
			// computing rhs3
			//---------------------
			for(int ii=0; ii<3; ii++)
			{
				//-------------		
				// z-dir
				//-------------
				REAL ss, Ak2;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0, vc1, vc2;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);
				Ak2 = stencil_Ay[ii] / ss;

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
				vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
				
				fp_shared[4*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_v[ii] + E2P * vc1 + E3P * vc2);
				fm_shared[4*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_v[ii] + E2M * vc1 + E3M * vc2);
			}
			
			{
				// REAL rhs = rhs3_shared[threadIdx.x +16*(threadIdx.y+1) +64*(threadIdx.z-1)*flag];
				// REAL rhs = rhs3_shared[threadIdx.x +16*(threadIdx.y+1) +64*threadIdx.z];
				__syncthreads();

				{	
					// z_rhs3+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					tmp_r = OCFD_weno7_SYMBO_kernel_P_opt(&fp_shared[offset+1]); // without transferring data between threads.
					tmp_l = OCFD_weno7_SYMBO_kernel_P_opt(&fp_shared[offset]);
					rhs3 += -Ajacobi*(tmp_r - tmp_l)*hz_d_rcp; // put_du_p_kernel
					get_SoA(du, x-job.start.x-flag2, y-job.start.y, z-job.start.z, 1) = u_shared[ 1 +threadIdx.x +15*threadIdx.y +64*threadIdx.z - flag2 ]; // rhs2
				}
				
				{
					// z_rhs3-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					tmp_r = OCFD_weno7_SYMBO_kernel_M_opt(&fm_shared[offset+1]); // without transferring data between threads.
					tmp_l = OCFD_weno7_SYMBO_kernel_M_opt(&fm_shared[offset]);
					rhs3 += -Ajacobi*(tmp_r - tmp_l)*hz_d_rcp; // put_du_m_kernel
					v_shared[ ( 1 +threadIdx.x +15*threadIdx.y +64*threadIdx.z) *flag1 ] = rhs3;
					// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = rhs3;
				}
			}
			
			//---------------------
			// computing rhs4
			//---------------------
			for(int ii=0; ii<3; ii++)
			{
				//-------------		
				// z-dir
				//-------------
				REAL ss, Ak3;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0, wc1, wc2;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);
				Ak3 = stencil_Az[ii] / ss;

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
				wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3;
				
				fp_shared[4*ii +offset] = tmp0 * (split_C1_d * E1P * stencil_w[ii] + E2P * wc1 + E3P * wc2);
				fm_shared[4*ii +offset] = tmp0 * (split_C1_d * E1M * stencil_w[ii] + E2M * wc1 + E3M * wc2);
			}
			
			{
				__syncthreads();

				{	
					// z_rhs4+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					tmp_r = OCFD_weno7_SYMBO_kernel_P_opt(&fp_shared[offset+1]); // without transferring data between threads.
					tmp_l = OCFD_weno7_SYMBO_kernel_P_opt(&fp_shared[offset]);
					rhs4 += -Ajacobi*(tmp_r - tmp_l)*hz_d_rcp; // put_du_p_kernel
					get_SoA(du, x-job.start.x-flag2, y-job.start.y, z-job.start.z, 2) = v_shared[ 1 +threadIdx.x +15*threadIdx.y +64*threadIdx.z - flag2 ]; // rhs3
				}
				
				{
					// z_rhs4-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					tmp_r = OCFD_weno7_SYMBO_kernel_M_opt(&fm_shared[offset+1]); // without transferring data between threads.
					tmp_l = OCFD_weno7_SYMBO_kernel_M_opt(&fm_shared[offset]);
					rhs4 += -Ajacobi*(tmp_r - tmp_l)*hz_d_rcp; // put_du_m_kernel
					w_shared[ ( 1 +threadIdx.x +15*threadIdx.y +64*threadIdx.z) *flag1 ] = rhs4;
					// if(threadIdx.x != (blockDim.x-1)) get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = rhs4;
				}
			}
			
			//---------------------
			// computing rhs5
			//---------------------
			for(int ii=0; ii<3; ii++)
			{
				//-------------		
				// z-dir
				//-------------
				REAL ss, Ak1, Ak2, Ak3;
				REAL vs, E1, E2, E3, E1P, E2P, E3P, E1M, E2M, E3M;
				REAL tmp0, uc1, uc2, vc1, vc2, wc1, wc2, vvc1, vvc2, vv, W2;
				
				ss = sqrt(stencil_Ax[ii]*stencil_Ax[ii] + stencil_Ay[ii]*stencil_Ay[ii] + stencil_Az[ii]*stencil_Az[ii]);
				Ak1 = stencil_Ax[ii] / ss;
				Ak2 = stencil_Ay[ii] / ss;
				Ak3 = stencil_Az[ii] / ss;

				vs = stencil_Ax[ii] * stencil_u[ii] 
				   + stencil_Ay[ii] * stencil_v[ii]
				   + stencil_Az[ii] * stencil_w[ii];

				E1 = vs;
				E2 = vs - stencil_cc[ii] * ss;
				E3 = vs + stencil_cc[ii] * ss;

				E1P = (E1 + sqrt(E1 * E1 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E2P = (E2 + sqrt(E2 * E2 + epsl_sw_d * epsl_sw_d)) * 0.50;
				E3P = (E3 + sqrt(E3 * E3 + epsl_sw_d * epsl_sw_d)) * 0.50;

				E1M = E1 - E1P;
				E2M = E2 - E2P;
				E3M = E3 - E3P;
				// ----------------------------------------
				tmp0 = stencil_d[ii] *Gamma_d_rcp;
				uc1  = stencil_u[ii] - stencil_cc[ii] * Ak1;
				uc2  = stencil_u[ii] + stencil_cc[ii] * Ak1;
				vc1  = stencil_v[ii] - stencil_cc[ii] * Ak2;
				vc2  = stencil_v[ii] + stencil_cc[ii] * Ak2;
				wc1  = stencil_w[ii] - stencil_cc[ii] * Ak3;
				wc2  = stencil_w[ii] + stencil_cc[ii] * Ak3;
				vvc1 = (uc1 * uc1 + vc1 * vc1 + wc1 * wc1) * 0.50;
				vvc2 = (uc2 * uc2 + vc2 * vc2 + wc2 * wc2) * 0.50;
				vv = (Gamma_d - 1.0) * (stencil_u[ii] * stencil_u[ii] 
									  + stencil_v[ii] * stencil_v[ii]
									  + stencil_w[ii] * stencil_w[ii] );
				W2 = split_C3_d * stencil_cc[ii] * stencil_cc[ii];
				
				fp_shared[4*ii +offset] = tmp0 * (E1P * vv + E2P * vvc1 + E3P * vvc2 + W2 * (E2P + E3P));
				fm_shared[4*ii +offset] = tmp0 * (E1M * vv + E2M * vvc1 + E3M * vvc2 + W2 * (E2M + E3M));
			}
			
			{
				__syncthreads();

				{	
					// z_rhs5+
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					tmp_r = OCFD_weno7_SYMBO_kernel_P_opt(&fp_shared[offset+1]); // without transferring data between threads.
					tmp_l = OCFD_weno7_SYMBO_kernel_P_opt(&fp_shared[offset]);
					rhs5 += -Ajacobi*(tmp_r - tmp_l)*hz_d_rcp; // put_du_p_kernel
					get_SoA(du, x-job.start.x-flag2, y-job.start.y, z-job.start.z, 3) = w_shared[ 1 +threadIdx.x +15*threadIdx.y +64*threadIdx.z - flag2 ]; // rhs4
				}
				
				{
					// z_rhs5-
					REAL tmp_r = 0.0, tmp_l = 0.0;
					
					tmp_r = OCFD_weno7_SYMBO_kernel_M_opt(&fm_shared[offset+1]); // without transferring data between threads.
					tmp_l = OCFD_weno7_SYMBO_kernel_M_opt(&fm_shared[offset]);
					rhs5 += -Ajacobi*(tmp_r - tmp_l)*hz_d_rcp; // put_du_m_kernel
					cc_shared[ ( 1 +threadIdx.x +15*threadIdx.y +64*threadIdx.z) *flag1 ] = rhs5;
					__syncthreads();
					get_SoA(du, x-job.start.x-flag2, y-job.start.y, z-job.start.z, 4) = cc_shared[ 1 +threadIdx.x +15*threadIdx.y +64*threadIdx.z - flag2 ]; // rhs5
					// if(threadIdx.x != (blockDim.x-1)) get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = rhs5;
				}
			}
		}
	}
// */
		// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = rhs1;
		// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = rhs2;
		// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = rhs3;
		// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = rhs4;
		// get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = rhs5;
	/*
	//load rhs[5] parameters from local memory to  global memory 
	//
	// __syncthreads(); // ★ Very Important ★
	// if( threadIdx.x != (blockDim.x-1) && threadIdx.y != (blockDim.y-1) &&  threadIdx.z != (blockDim.z-1))
	if( threadIdx.x != (blockDim.x-1) && threadIdx.y != (blockDim.y-1) )
	{
		
		get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 0) = rhs1_shared[threadIdx.x+16*threadIdx.y+64*threadIdx.z];
		get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 1) = rhs2_shared[threadIdx.x+16*threadIdx.y+64*threadIdx.z];
		get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 2) = rhs3_shared[threadIdx.x+16*threadIdx.y+64*threadIdx.z];
		get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 3) = rhs4_shared[threadIdx.x+16*threadIdx.y+64*threadIdx.z];
		get_SoA(du, x-job.start.x, y-job.start.y, z-job.start.z, 4) = rhs5_shared[threadIdx.x+16*threadIdx.y+64*threadIdx.z];
	}
	*/
}


#ifdef __cplusplus
}
#endif
