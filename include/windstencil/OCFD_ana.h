#ifndef __OCFD_ANA_H
#define __OCFD_ANA_H
#include "cuda_commen.h"

#ifdef __cplusplus
extern "C"{
#endif

void ana_residual(cudaField PE_d, REAL *E0);
void ana_Jac();
void OCFD_ana(int style, int ID);
void ana_NAN_and_NT();
void init_time_average();

void get_inner(cudaField x1, cudaField x2);

#ifdef __cplusplus
}
#endif
#endif