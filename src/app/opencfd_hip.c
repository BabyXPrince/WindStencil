//----------------------------------------------------------------------------------------------------------------------------------------   
// OpenCFD-SC  ,  3-D compressible Navier-Stokes Finite difference Solver 
// Copyright by LI Xinliang, LHD, Institute of Mechanics, CAS, Email: lixl@imech.ac.cn
//  
// The default code is double precision computation
// If you want to use SINGLE PRECISION computation, you can change   "OCFD_REAL_KIND=8"  to "OCFD_REAL_KIND=4" ,
// and  "OCFD_DATA_TYPE=MPI_DOUBLE_PRECISION" to "OCFD_DATA_TYPE=MPI_REAL" in the file OpenCFD.h 
//---------------------------------------------------------------------------------------------------------------------------------------------- 
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"

#include "utility.h"
#include "parameters.h"

#include "OCFD_NS_Solver.h"
#include "OCFD_mpi.h"
#include "OCFD_init.h"
#include "cuda_commen.h"
#include "OCFD_mpi_dev.h"
#include "OCFD_filtering.h"

// 【Liu Version】

#ifdef __cplusplus
extern "C"{
#endif

int main(int argc, char *argv[]){
	
			REAL tstart0,tend0;
			tstart0 = MPI_Wtime();
			
			// REAL tstart1,tend1;
			// tstart1 = MPI_Wtime();
			
    mpi_init(&argc , &argv);

    read_parameters();

    opencfd_mem_init_mpi();  

    part();

    set_para_filtering();
    
    opencfd_mem_init_all();

    cuda_commen_init();
	
			// tend1 = MPI_Wtime();
			// if(my_id == 0) printf("Total S1 time is %lf s.\n" , (tend1 - tstart1)*1 );
			
			// REAL tstart2,tend2;
			// tstart2 = MPI_Wtime();

    NS_solver_real();
	
			// tend2 = MPI_Wtime();
			// if(my_id == 0) printf("Total S2 time is %lf s.\n" , (tend2 - tstart2)*1 );
			
			// REAL tstart3,tend3;
			// tstart3 = MPI_Wtime();

    opencfd_mem_finalize_all();

    mpi_finalize();
	
			// tend3 = MPI_Wtime();
			// if(my_id == 0) printf("Total S3 time is %lf s.\n" , (tend3 - tstart3)*1 );
			
			tend0 = MPI_Wtime();
			if(my_id == 0) printf("the Total OpenCFD time is %lf s.\n" , (tend0 - tstart0)*1 );
    
    return 0;
}

#ifdef __cplusplus
}
#endif

