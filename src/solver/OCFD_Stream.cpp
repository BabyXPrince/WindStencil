#include <stdio.h>
#include <stdlib.h>

#include "OCFD_Stream.h"
#include "OCFD_NS_Jacobian3d.h"
#include "parameters.h"
#include "OCFD_mpi_dev.h"
#include "parameters_d.h"
#include "commen_kernel.h"
#include "OCFD_Schemes_hybrid_auto.h"
#ifdef __cplusplus
extern "C" {
#endif

//static hipStream_t Stream[15];

void opencfd_mem_init_Stream(){
    for (int i = 0; i < 3; i++) hipStreamCreate(&Stream[i]);
    for (int i = 0; i < 3; i++) hipEventCreate(&Event[i]);
}

void opencfd_mem_finalize_Stream(){
    for (int i = 0; i < 3; ++i) hipStreamDestroy(Stream[i]);
    for (int i = 0; i < 3; ++i) hipEventDestroy(Event[i]);
}

void du_comput(int KRK){
	//pthread_create(&thread_handles[0], NULL, du_invis_Jacobian3d_inner, NULL);
	//pthread_create(&thread_handles[1], NULL, du_vis_Jacobian3d_outer, NULL);

	//for(int thread = 0; thread < 2; thread++)
	//	pthread_join(thread_handles[thread], NULL);
	if(IFLAG_HybridAuto == 1 && KRK == 1) Set_Scheme_HybridAuto(&Stream[0]);

	cuda_mem_value_init_warp(0.0 ,pdu_d->ptr, pdu_d->pitch, nx, ny, nz*5);

	switch(Stream_MODE){
        case 0://Non-stream
	    du_invis_Jacobian3d(NULL);
	    du_vis_Jacobian3d(NULL);
        break;

        case 1://launch: first invis, then vis
        du_invis_Jacobian3d_all(NULL);
	    du_vis_Jacobian3d_all(NULL);
		break;
		
		default: 
		if(my_id == 0) printf("\033[31mWrong Stream Mode! Please choose 0 or 1, 0:non stream; 1:stream\033[0m\n");
    }

}


void* du_invis_Jacobian3d_all(void* pthread_id){

	cudaJobPackage job(dim3(2*LAP, 2*LAP, 2*LAP), dim3(nx, ny, nz));
	du_invis_Jacobian3d_init(job, &Stream[0]);

	job.setup(dim3(3*LAP, 3*LAP, 3*LAP), dim3(nx-LAP, ny-LAP, nz-LAP));
    //direction X ------------------------------
	
	// du_invis_Jacobian3d_x(job, pfp_d, pfm_d, &Stream[0]);
	du_invis_Jacobian3d_outer_x(&Stream[1]);

    //direction Y ------------------------------

	hipEventRecord(Event[0], Stream[0]);
	hipEventRecord(Event[1], Stream[1]);
	hipStreamWaitEvent(Stream[0], Event[1], 0);
	// du_invis_Jacobian3d_y(job, pfp_d, pfm_d, &Stream[0]);
	du_invis_Jacobian3d_outer_y(&Stream[1], &Event[0]);

    //direction Z ------------------------------

	hipEventRecord(Event[0], Stream[0]);
	hipEventRecord(Event[1], Stream[1]);
	hipStreamWaitEvent(Stream[0], Event[1], 0);
	// du_invis_Jacobian3d_z(job, pfp_d, pfm_d, &Stream[0]);
	du_invis_Jacobian3d_outer_z(&Stream[1], &Event[0]);
	hipEventRecord(Event[1], Stream[1]);

	return NULL;
}

void* du_vis_Jacobian3d_all(void* pthread_id){

	hipStreamWaitEvent(Stream[2], Event[1], 0);
	du_viscous_Jacobian3d_init(&Stream[2]);

    //direction X ------------------------------
	
	du_viscous_Jacobian3d_x_init(&Stream[2]);
	hipEventRecord(Event[2], Stream[2]);
	du_vis_Jacobian3d_inner_x(&Stream[2]);
	hipStreamWaitEvent(Stream[1], Event[2], 0);
	du_vis_Jacobian3d_outer_x(&Stream[1]);

    //direction Y ------------------------------

	hipEventRecord(Event[2], Stream[1]);
	hipStreamWaitEvent(Stream[2], Event[2], 0);
	du_viscous_Jacobian3d_y_init(&Stream[2]);
	hipEventRecord(Event[2], Stream[2]);
	du_vis_Jacobian3d_inner_y(&Stream[2]);
	hipStreamWaitEvent(Stream[1], Event[2], 0);
	du_vis_Jacobian3d_outer_y(&Stream[1]);

    //direction X ------------------------------
	
	hipEventRecord(Event[2], Stream[1]);
	hipStreamWaitEvent(Stream[2], Event[2], 0);
	du_viscous_Jacobian3d_z_init(&Stream[2]);
	hipEventRecord(Event[2], Stream[2]);
	du_vis_Jacobian3d_inner_z(&Stream[2]);
	hipStreamWaitEvent(Stream[1], Event[2], 0);
	du_vis_Jacobian3d_outer_z(&Stream[1]);
	

	return NULL;
}
//void* du_vis_Jacobian3d_all(void* pthread_id){
//
//	du_viscous_Jacobian3d_init(&Stream[2]);
//
//    //direction X ------------------------------
//	
//	du_viscous_Jacobian3d_x_init(&Stream[2]);
//	hipEventRecord(Event[2], Stream[2]);
//	du_vis_Jacobian3d_inner_x(&Stream[2]);
//	hipStreamWaitEvent(Stream[3], Event[1], 0);
//	hipStreamWaitEvent(Stream[3], Event[2], 0);
//	du_vis_Jacobian3d_outer_x(&Stream[3]);
//
//    //direction Y ------------------------------
//
//	hipEventRecord(Event[2], Stream[3]);
//	hipStreamWaitEvent(Stream[2], Event[2], 0);
//	du_viscous_Jacobian3d_y_init(&Stream[2]);
//	hipEventRecord(Event[2], Stream[2]);
//	du_vis_Jacobian3d_inner_y(&Stream[2]);
//	hipStreamWaitEvent(Stream[3], Event[2], 0);
//	du_vis_Jacobian3d_outer_y(&Stream[3]);
//
//    //direction X ------------------------------
//	
//	hipEventRecord(Event[2], Stream[3]);
//	hipStreamWaitEvent(Stream[2], Event[2], 0);
//	du_viscous_Jacobian3d_z_init(&Stream[2]);
//	hipEventRecord(Event[2], Stream[2]);
//	du_vis_Jacobian3d_inner_z(&Stream[2]);
//	hipStreamWaitEvent(Stream[3], Event[2], 0);
//	du_vis_Jacobian3d_outer_z(&Stream[3]);
//	
//
//	return NULL;
//}


void* du_vis_Jacobian3d_inner_x(hipStream_t *stream){

	cudaJobPackage job(dim3(3*LAP, 3*LAP, 3*LAP), dim3(nx-LAP, ny-LAP, nz-LAP));
	du_viscous_Jacobian3d_x_final(job, stream);

	return NULL;
}

void* du_vis_Jacobian3d_inner_y(hipStream_t *stream){

	cudaJobPackage job(dim3(3*LAP, 3*LAP, 3*LAP), dim3(nx-LAP, ny-LAP, nz-LAP));
	du_viscous_Jacobian3d_y_final(job, stream);

	return NULL;
}

void* du_vis_Jacobian3d_inner_z(hipStream_t *stream){

	cudaJobPackage job(dim3(3*LAP, 3*LAP, 3*LAP), dim3(nx-LAP, ny-LAP, nz-LAP));
	du_viscous_Jacobian3d_z_final(job, stream);

	return NULL;
}


void* du_invis_Jacobian3d(void* pthread_id){

	exchange_boundary_xyz_packed_dev(pd , pd_d);
	exchange_boundary_xyz_packed_dev(pu , pu_d);
	exchange_boundary_xyz_packed_dev(pv , pv_d); 
	exchange_boundary_xyz_packed_dev(pw , pw_d);
	exchange_boundary_xyz_packed_dev(pT , pT_d);

	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, nz_lap));
	du_invis_Jacobian3d_init(job, &Stream[0]);
	
			// du_invis_Jacobian3d_fpY(job, pfp_d, pfm_d, &Stream[0]);
			// du_invis_Jacobian3d_fpZ(job, pfp_d, pfm_d, &Stream[0]);
			du_invis_Jacobian3d_x(job, pfp_d, pfm_d, &Stream[0]);
			
			// check vis: rhs1 ~ rhs5.
				// double *rhs_test =(double*)calloc( (5*nz*ny*(pdu_d->pitch)), sizeof(double) );
				// hipMemcpy(rhs_test, pdu_d->ptr, 5*nz*ny*(pdu_d->pitch)*sizeof(double), hipMemcpyDeviceToHost);
			
				// FILE *file_index = fopen( "rhs_mix.txt" , "w");
				
				// for (int k = 0; k < nz; k++) {
				// for (int j = 0; j < ny; j++) {
				// for (int i = 0; i < nx; i++) {
				
					// int offset1 = i + j* (pdu_d->pitch) + k *ny*(pdu_d->pitch);
					// int offset2 = (pdu_d->pitch)*ny*nz;
					
					// fprintf(file_index , "%d, %d, %d, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E\n",
										// i, j, k, rhs_test[offset1+offset2*0], rhs_test[offset1+offset2*1], rhs_test[offset1+offset2*2], rhs_test[offset1+offset2*3], rhs_test[offset1+offset2*4] );
				// }
				// }
				// }
				// fclose(file_index);
				
				printf("============= du_invis_Jacobian3d Done. =============\n");
	
	// du_invis_Jacobian3d_x(job, pfp_d, pfm_d, &Stream[0]);
	// du_invis_Jacobian3d_y(job, pfp_d, pfm_d, &Stream[0]);
	// du_invis_Jacobian3d_z(job, pfp_d, pfm_d, &Stream[0]);

	return NULL;
}

void* du_vis_Jacobian3d(void* pthread_id){

	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, nz_lap));
	du_viscous_Jacobian3d_init(&Stream[0]);
	
			// check X: puk_d, pvk_d, pwk_d, pTk_d
				// double *puk_d_test =(double*)calloc( (nz*ny*(puk_d->pitch)), sizeof(double) );
				// double *pvk_d_test =(double*)calloc( (nz*ny*(pvk_d->pitch)), sizeof(double) );
				// double *pwk_d_test =(double*)calloc( (nz*ny*(pwk_d->pitch)), sizeof(double) );
				// double *pTk_d_test =(double*)calloc( (nz*ny*(pTk_d->pitch)), sizeof(double) );
				// hipMemcpy(puk_d_test, puk_d->ptr, nz*ny*(puk_d->pitch)*sizeof(double), hipMemcpyDeviceToHost);
				// hipMemcpy(pvk_d_test, pvk_d->ptr, nz*ny*(pvk_d->pitch)*sizeof(double), hipMemcpyDeviceToHost);
				// hipMemcpy(pwk_d_test, pwk_d->ptr, nz*ny*(pwk_d->pitch)*sizeof(double), hipMemcpyDeviceToHost);
				// hipMemcpy(pTk_d_test, pTk_d->ptr, nz*ny*(pTk_d->pitch)*sizeof(double), hipMemcpyDeviceToHost);
			
				// FILE *file_index_uk = fopen( "uk_mix.txt" , "w");
				
				// for (int k = 0; k < nz; k++) {
				// for (int j = 0; j < ny; j++) {
				// for (int i = 0; i < nx; i++) {
				
					// int offset = i + j* (puk_d->pitch) + k *ny*(puk_d->pitch); // get_Field(pfy, x-LAP, y-LAP, z-LAP) = tmp/hx_d
					// fprintf(file_index_uk , "%d, %d, %d, [%d], %24.16E, %24.16E, %24.16E, %24.16E\n",
										// i, j, k, offset, puk_d_test[offset], pvk_d_test[offset], pwk_d_test[offset], pTk_d_test[offset] );
				// }
				// }
				// }
				// fclose(file_index_uk);
	
	du_viscous_Jacobian3d_x_init(&Stream[0]);
	
			// check X: pEv1_d, pEv2_d, pEv3_d, pEv4_d.  // [nz+2*LAP][ny+2*LAP][nx+2*LAP]
				// int szx=nx+2*LAP, szy=ny+2*LAP, szz=nz+2*LAP;
				// double *pEv1_d_test =(double*)calloc( (szz*szy*(pEv1_d->pitch)), sizeof(double) );
				// double *pEv2_d_test =(double*)calloc( (szz*szy*(pEv2_d->pitch)), sizeof(double) );
				// double *pEv3_d_test =(double*)calloc( (szz*szy*(pEv3_d->pitch)), sizeof(double) );
				// double *pEv4_d_test =(double*)calloc( (szz*szy*(pEv4_d->pitch)), sizeof(double) );
				// hipMemcpy(pEv1_d_test, pEv1_d->ptr, szz*szy*(pEv1_d->pitch)*sizeof(double), hipMemcpyDeviceToHost);
				// hipMemcpy(pEv2_d_test, pEv2_d->ptr, szz*szy*(pEv2_d->pitch)*sizeof(double), hipMemcpyDeviceToHost);
				// hipMemcpy(pEv3_d_test, pEv3_d->ptr, szz*szy*(pEv3_d->pitch)*sizeof(double), hipMemcpyDeviceToHost);
				// hipMemcpy(pEv4_d_test, pEv4_d->ptr, szz*szy*(pEv4_d->pitch)*sizeof(double), hipMemcpyDeviceToHost);
			
				// FILE *file_index_pEv = fopen( "pEv_mix.txt" , "w");
				
				// for (int k = 0; k < szz; k++) {
				// for (int j = 0; j < szy; j++) {
				// for (int i = 0; i < szx; i++) {
				
					// int offset = i + j* (pEv1_d->pitch) + k *szy*(pEv1_d->pitch); // get_Field_LAP(Ev1 , x,y,z) = ( akx*s11 + aky*s12 + akz*s13 );
					// fprintf(file_index_pEv , "%d, %d, %d, [%d], %24.16E, %24.16E, %24.16E, %24.16E\n",
										// i, j, k, offset, pEv1_d_test[offset], pEv2_d_test[offset], pEv3_d_test[offset], pEv4_d_test[offset] );
				// }
				// }
				// }
				// fclose(file_index_pEv);
				
	exchange_boundary_x_packed_dev(pEv1 , pEv1_d , Iperiodic[0]);
    exchange_boundary_x_packed_dev(pEv2 , pEv2_d , Iperiodic[0]);
    exchange_boundary_x_packed_dev(pEv3 , pEv3_d , Iperiodic[0]);
    exchange_boundary_x_packed_dev(pEv4 , pEv4_d , Iperiodic[0]);
	du_viscous_Jacobian3d_x_final(job, &Stream[0]);
	
			// check X vis: rhs1 ~ rhs5.
				// double *rhs_test =(double*)calloc( (5*nz*ny*(pdu_d->pitch)), sizeof(double) );
				// hipMemcpy(rhs_test, pdu_d->ptr, 5*nz*ny*(pdu_d->pitch)*sizeof(double), hipMemcpyDeviceToHost);
			
				// FILE *file_index_rhsx = fopen( "rhs_visx_mix.txt" , "w");
				
				// for (int k = 0; k < nz; k++) {
				// for (int j = 0; j < ny; j++) {
				// for (int i = 0; i < nx; i++) {
				
					// int offset1 = i + j* (pdu_d->pitch) + k *ny*(pdu_d->pitch);
					// int offset2 = (pdu_d->pitch)*ny*nz;
					
					// fprintf(file_index_rhsx , "%d, %d, %d, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E\n",
										// i, j, k, rhs_test[offset1+offset2*0], rhs_test[offset1+offset2*1], rhs_test[offset1+offset2*2], rhs_test[offset1+offset2*3], rhs_test[offset1+offset2*4] );
				// }
				// }
				// }
				// fclose(file_index_rhsx);
				
	du_viscous_Jacobian3d_y_init(&Stream[0]);
	exchange_boundary_y_packed_dev(pEv1 , pEv1_d , Iperiodic[1]);
    exchange_boundary_y_packed_dev(pEv2 , pEv2_d , Iperiodic[1]);
    exchange_boundary_y_packed_dev(pEv3 , pEv3_d , Iperiodic[1]);
	exchange_boundary_y_packed_dev(pEv4 , pEv4_d , Iperiodic[1]);
	boundary_symmetry_pole_vis_y(&Stream[0]);
	du_viscous_Jacobian3d_y_final(job, &Stream[0]);

	du_viscous_Jacobian3d_z_init(&Stream[0]);
	exchange_boundary_z_packed_dev(pEv1 , pEv1_d ,Iperiodic[2]);
    exchange_boundary_z_packed_dev(pEv2 , pEv2_d ,Iperiodic[2]);
    exchange_boundary_z_packed_dev(pEv3 , pEv3_d ,Iperiodic[2]);
    exchange_boundary_z_packed_dev(pEv4 , pEv4_d ,Iperiodic[2]);
	du_viscous_Jacobian3d_z_final(job, &Stream[0]);
	
			// check Z vis: rhs1 ~ rhs5.
				// double *rhs_test =(double*)calloc( (5*nz*ny*(pdu_d->pitch)), sizeof(double) );
				// hipMemcpy(rhs_test, pdu_d->ptr, 5*nz*ny*(pdu_d->pitch)*sizeof(double), hipMemcpyDeviceToHost);
			
				// FILE *file_index_rhsz = fopen( "rhs_visz_mix.txt" , "w");
				
				// for (int k = 0; k < nz; k++) {
				// for (int j = 0; j < ny; j++) {
				// for (int i = 0; i < nx; i++) {
				
					// int offset1 = i + j* (pdu_d->pitch) + k *ny*(pdu_d->pitch);
					// int offset2 = (pdu_d->pitch)*ny*nz;
					
					// fprintf(file_index_rhsz , "%d, %d, %d, %24.16E, %24.16E, %24.16E, %24.16E, %24.16E\n",
										// i, j, k, rhs_test[offset1+offset2*0], rhs_test[offset1+offset2*1], rhs_test[offset1+offset2*2], rhs_test[offset1+offset2*3], rhs_test[offset1+offset2*4] );
				// }
				// }
				// }
				// fclose(file_index_rhsz);

	return NULL;
}

void* du_invis_Jacobian3d_outer_init_x(hipStream_t *stream){
//-------------x outer p init----------------
	cudaJobPackage job(dim3(LAP, 2*LAP, 2*LAP), dim3(3*LAP, ny, nz));
	du_invis_Jacobian3d_init(job, stream);
//-------------x outer m init----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	du_invis_Jacobian3d_init(job, stream);
	
	return NULL;
}


void* du_invis_Jacobian3d_outer_x_x(hipStream_t *stream){
//-------------x outer p x----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	// du_invis_Jacobian3d_x(job, pfp_d, pfm_d, stream);
//-------------x outer m x----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	// du_invis_Jacobian3d_x(job, pfp_d, pfm_d, stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_y_x(hipStream_t *stream){
//-------------x outer p y----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	// du_invis_Jacobian3d_y(job, pfp_d, pfm_d, stream);
//-------------x outer m y----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	// du_invis_Jacobian3d_y(job, pfp_d, pfm_d, stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_z_x(hipStream_t *stream){
//-------------x outer p z----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	// du_invis_Jacobian3d_z(job, pfp_d, pfm_d, stream);
//-------------x outer m z----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	// du_invis_Jacobian3d_z(job, pfp_d, pfm_d, stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_init_y(hipStream_t *stream){
//-------------y outer p init----------------
	cudaJobPackage job(dim3(2*LAP, LAP, 2*LAP), dim3(nx, 3*LAP, nz));
	du_invis_Jacobian3d_init(job, stream);
//-------------y outer m init----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	du_invis_Jacobian3d_init(job, stream);
		
	return NULL;
}
	

void* du_invis_Jacobian3d_outer_x_y(hipStream_t *stream){
//-------------y outer p x----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	// du_invis_Jacobian3d_x(job, pfp_d, pfm_d, stream);
//-------------y outer m x----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	// du_invis_Jacobian3d_x(job, pfp_d, pfm_d, stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_y_y(hipStream_t *stream){
//-------------y outer p y----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	// du_invis_Jacobian3d_y(job, pfp_d, pfm_d, stream);
//-------------y outer m y----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	// du_invis_Jacobian3d_y(job, pfp_d, pfm_d, stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_z_y(hipStream_t *stream){
//-------------y outer p z----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	// du_invis_Jacobian3d_z(job, pfp_d, pfm_d, stream);
//-------------y outer m z----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	// du_invis_Jacobian3d_z(job, pfp_d, pfm_d, stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_init_z(hipStream_t *stream){
//-------------z outer p init----------------
	cudaJobPackage job(dim3(2*LAP, 2*LAP, LAP), dim3(nx, ny, 3*LAP));
	du_invis_Jacobian3d_init(job, stream);
//-------------z outer m init----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	du_invis_Jacobian3d_init(job, stream);

	return NULL;
}

void* du_invis_Jacobian3d_outer_x_z(hipStream_t *stream){
//-------------z outer p x----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	// du_invis_Jacobian3d_x(job, pfp_d, pfm_d, stream);
//-------------z outer m x----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	// du_invis_Jacobian3d_x(job, pfp_d, pfm_d, stream);

	return NULL;
}

void* du_invis_Jacobian3d_outer_y_z(hipStream_t *stream){
//-------------z outer p----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	// du_invis_Jacobian3d_y(job, pfp_d, pfm_d, stream);
//-------------z outer m----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	// du_invis_Jacobian3d_y(job, pfp_d, pfm_d, stream);

	return NULL;
}

void* du_invis_Jacobian3d_outer_z_z(hipStream_t *stream){
//-------------z outer p----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	// du_invis_Jacobian3d_z(job, pfp_d, pfm_d, stream);
//-------------z outer m----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	// du_invis_Jacobian3d_z(job, pfp_d, pfm_d, stream);

	return NULL;
}

void* du_invis_Jacobian3d_outer_exchange(hipStream_t *stream){

	exchange_boundary_xyz_Async_packed_dev(pd , pd_d , stream);
	exchange_boundary_xyz_Async_packed_dev(pu , pu_d , stream);
	exchange_boundary_xyz_Async_packed_dev(pv , pv_d , stream); 
	exchange_boundary_xyz_Async_packed_dev(pw , pw_d , stream);
	exchange_boundary_xyz_Async_packed_dev(pT , pT_d , stream);
	
	return NULL;
}

void* du_invis_Jacobian3d_outer_x(hipStream_t *stream){

	exchange_boundary_x_Async_packed_dev(pd , pd_d , Iperiodic[0] , stream);
	exchange_boundary_x_Async_packed_dev(pu , pu_d , Iperiodic[0] , stream);
	exchange_boundary_x_Async_packed_dev(pv , pv_d , Iperiodic[0] , stream); 
	exchange_boundary_x_Async_packed_dev(pw , pw_d , Iperiodic[0] , stream);
	exchange_boundary_x_Async_packed_dev(pT , pT_d , Iperiodic[0] , stream);

	du_invis_Jacobian3d_outer_init_x(stream);
	du_invis_Jacobian3d_outer_x_x(stream);
	du_invis_Jacobian3d_outer_x_y(stream);
	du_invis_Jacobian3d_outer_x_z(stream);
	
	return NULL;
}


void* du_invis_Jacobian3d_outer_y(hipStream_t *stream, hipEvent_t *event){
	
	exchange_boundary_y_Async_packed_dev(pd , pd_d , Iperiodic[1] , stream);
	exchange_boundary_y_Async_packed_dev(pu , pu_d , Iperiodic[1] , stream);
	exchange_boundary_y_Async_packed_dev(pv , pv_d , Iperiodic[1] , stream); 
	exchange_boundary_y_Async_packed_dev(pw , pw_d , Iperiodic[1] , stream);
	exchange_boundary_y_Async_packed_dev(pT , pT_d , Iperiodic[1] , stream);

	hipStreamWaitEvent(*stream, *event, 0);

	du_invis_Jacobian3d_outer_init_y(stream);
	du_invis_Jacobian3d_outer_y_x(stream);
	du_invis_Jacobian3d_outer_y_y(stream);
	du_invis_Jacobian3d_outer_y_z(stream);
	
	return NULL;
}


void* du_invis_Jacobian3d_outer_z(hipStream_t *stream, hipEvent_t *event){
	
	exchange_boundary_z_Async_packed_dev(pd , pd_d , Iperiodic[2] , stream);
	exchange_boundary_z_Async_packed_dev(pu , pu_d , Iperiodic[2] , stream);
	exchange_boundary_z_Async_packed_dev(pv , pv_d , Iperiodic[2] , stream); 
	exchange_boundary_z_Async_packed_dev(pw , pw_d , Iperiodic[2] , stream);
	exchange_boundary_z_Async_packed_dev(pT , pT_d , Iperiodic[2] , stream);

	hipStreamWaitEvent(*stream, *event, 0);

	du_invis_Jacobian3d_outer_init_z(stream);
	du_invis_Jacobian3d_outer_z_x(stream);
	du_invis_Jacobian3d_outer_z_y(stream);
	du_invis_Jacobian3d_outer_z_z(stream);
	
	return NULL;
}


void* du_vis_Jacobian3d_outer_x_x(hipStream_t *stream){

//-------------x outer p x----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	du_viscous_Jacobian3d_x_final(job, stream);
//-------------x outer m x----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	du_viscous_Jacobian3d_x_final(job, stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_y_x(hipStream_t *stream){
//-------------x outer p y----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	du_viscous_Jacobian3d_y_final(job, stream);
//-------------x outer m y----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	du_viscous_Jacobian3d_y_final(job, stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_z_x(hipStream_t *stream){
//-------------x outer p z----------------
	cudaJobPackage job(dim3(LAP, 3*LAP, 3*LAP), dim3(3*LAP, ny-LAP, nz-LAP));
	du_viscous_Jacobian3d_z_final(job, stream);
//-------------x outer m z----------------
	job.start.x = nx-LAP;
	job.end.x = nx_lap;
	du_viscous_Jacobian3d_z_final(job, stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_x_y(hipStream_t *stream){
//-------------y outer p x----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	du_viscous_Jacobian3d_x_final(job, stream);
//-------------y outer m x----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	du_viscous_Jacobian3d_x_final(job, stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_y_y(hipStream_t *stream){
//-------------y outer p y----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	du_viscous_Jacobian3d_y_final(job, stream);
//-------------y outer m y----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	du_viscous_Jacobian3d_y_final(job, stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_z_y(hipStream_t *stream){
//-------------y outer p z----------------
	cudaJobPackage job(dim3(LAP, LAP, 3*LAP), dim3(nx_lap, 3*LAP, nz-LAP));
	du_viscous_Jacobian3d_z_final(job, stream);
//-------------y outer m z----------------
	job.start.y = ny-LAP;
	job.end.y = ny_lap;
	du_viscous_Jacobian3d_z_final(job, stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_x_z(hipStream_t *stream){
//-------------z outer p x----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	du_viscous_Jacobian3d_x_final(job, stream);
//-------------z outer m x----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	du_viscous_Jacobian3d_x_final(job, stream);

	return NULL;
}

void* du_vis_Jacobian3d_outer_y_z(hipStream_t *stream){
//-------------z outer p y----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	du_viscous_Jacobian3d_y_final(job, stream);
//-------------z outer m y----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	du_viscous_Jacobian3d_y_final(job, stream);

	return NULL;
}

void* du_vis_Jacobian3d_outer_z_z(hipStream_t *stream){
//-------------z outer p z----------------
	cudaJobPackage job(dim3(LAP, LAP, LAP), dim3(nx_lap, ny_lap, 3*LAP));
	du_viscous_Jacobian3d_z_final(job, stream);
//-------------z outer m z----------------
	job.start.z = nz-LAP;
	job.end.z = nz_lap;
	du_viscous_Jacobian3d_z_final(job, stream);

	return NULL;
}

void* du_vis_Jacobian3d_outer_x(hipStream_t *stream){

	exchange_boundary_x_Async_packed_dev(pEv1 , pEv1_d , Iperiodic[0], stream);
    exchange_boundary_x_Async_packed_dev(pEv2 , pEv2_d , Iperiodic[0], stream);
    exchange_boundary_x_Async_packed_dev(pEv3 , pEv3_d , Iperiodic[0], stream);
    exchange_boundary_x_Async_packed_dev(pEv4 , pEv4_d , Iperiodic[0], stream);

	du_vis_Jacobian3d_outer_x_x(stream);
	du_vis_Jacobian3d_outer_x_y(stream);
	du_vis_Jacobian3d_outer_x_z(stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_y(hipStream_t *stream){

	exchange_boundary_y_Async_packed_dev(pEv1 , pEv1_d , Iperiodic[1], stream);
    exchange_boundary_y_Async_packed_dev(pEv2 , pEv2_d , Iperiodic[1], stream);
    exchange_boundary_y_Async_packed_dev(pEv3 , pEv3_d , Iperiodic[1], stream);
	exchange_boundary_y_Async_packed_dev(pEv4 , pEv4_d , Iperiodic[1], stream);
	
	boundary_symmetry_pole_vis_y(stream);

	du_vis_Jacobian3d_outer_y_x(stream);
	du_vis_Jacobian3d_outer_y_y(stream);
	du_vis_Jacobian3d_outer_y_z(stream);
	
	return NULL;
}

void* du_vis_Jacobian3d_outer_z(hipStream_t *stream){

	exchange_boundary_z_Async_packed_dev(pEv1 , pEv1_d , Iperiodic[2], stream);
    exchange_boundary_z_Async_packed_dev(pEv2 , pEv2_d , Iperiodic[2], stream);
    exchange_boundary_z_Async_packed_dev(pEv3 , pEv3_d , Iperiodic[2], stream);
    exchange_boundary_z_Async_packed_dev(pEv4 , pEv4_d , Iperiodic[2], stream);

	du_vis_Jacobian3d_outer_z_x(stream);
	du_vis_Jacobian3d_outer_z_y(stream);
	du_vis_Jacobian3d_outer_z_z(stream);

	return NULL;
}

#ifdef __cplusplus
}
#endif
