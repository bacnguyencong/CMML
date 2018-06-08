#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <time.h>

#include "mylib.h"
#include "mex.h"

#define EXACTINPUTS  5
#define EXACTOUTPUTS 2


/* Input Arguments */
#define	INPUT_X         prhs[0] 
#define	INPUT_T         prhs[1] 
#define	INPUT_M0        prhs[2] 
#define	INPUT_beta      prhs[3] 
#define	INPUT_max_iters prhs[4] 

/* Output Arguments */

#define	OUTPUT_M    	plhs[0]
#define	OUTPUT_C    	plhs[1]

/*------------------------------------------------------------------------*/

FILE * fp;

double frobenius_norm(int d, double **M){
    double ret = 0;
    int i, j;
    for (i = 0; i < d; ++ i)
        for (j = 0; j < d; ++ j)
            ret += M[i][j]*M[i][j];
    return ret;
}

void add_MM(int d, double ** M, R_VEC pM0, double c){
    int i, j;
    for (i = 0; i < d; ++ i)
        for (j = 0; j < d; ++ j)
            M[i][j] += pM0[i*d+j] * c;
}
    

double getCostFunct(int d, R_VEC pX, int m, I_VEC pT, R_VEC pM0, double beta_in, double **M){
    
    int t, i, j, l;
    double cost = 0, dj, dl;
    
    for (t = 0; t < m; ++ t){
        i = pT[t*3]     - 1;
        j = pT[t*3 + 1] - 1;
        l = pT[t*3 + 2] - 1;
        dj = distanceVV(d, M, pX + d*i,  pX + d*j);
        dl = distanceVV(d, M, pX + d*i,  pX + d*l);
            
        /* if it is a violated example */
        if (dj + 1 > dl)
            cost += dj + 1 - dl;
    }
        
    cost = cost/m + beta_in*0.5*frobenius_norm(d, M);
    
    return cost;
}


double distanceMetricLearning(int d, R_VEC pX, int m, I_VEC pT, R_VEC pM0, double beta_in, 
        double **M, int MAX_ITERS){
       
    R_VEC First, Second;
    I_VEC order;

    R_VEC  temp, alpha, beta, v;
    R_MAT  Q, Z, tempM;

    double dj, dl, cost, eta, pre_cost = 1e99;
    int i, j, l, t, k, ii, jj;
    int  iters, epoch;

    /* initial parameters */
    k = MAX_LACZOS;
    if ( d <= MAX_LACZOS )
    	k = d < 15 ? d : 15;

    order    = createIV(m);
    temp     = createRV(d);
    First    = createRV(d);
    Second   = createRV(d);
    v        = createRV(d);
    alpha    = createRV(k + 1);
    beta     = createRV(k + 1);
    Q 	     = createRM(d, d);
    Z 	     = createRM(k + 1, k + 1);
    tempM    = createRM(d, d);

    /* the initial solution */
    for (i = 0; i < d; ++ i)
        for (j = 0; j < d; ++ j)
            M[i][j] = pM0[i*d + j];
    
    for (i = 0; i < m; ++ i)
    	order[i] = i;

    /* random the order of stochastic gradient */
    randperm(m, order);

    /* begin algorithm */
    for (epoch = 1, iters = 1; epoch <= MAX_EPOCH; ++ epoch){
        
        /* run new epoch */
        for (t = 0; (t < m) && (iters <= MAX_ITERS); ++ t, ++ iters){

            i = pT[order[t]*3]     - 1;
            j = pT[order[t]*3 + 1] - 1;
            l = pT[order[t]*3 + 2] - 1;
                 
            dj = distanceVV(d, M, pX + d*i, pX + d*j);
            dl = distanceVV(d, M, pX + d*i, pX + d*l);
            
            /* if it is a violated example */
            if (dj + 1 > dl){               
                /* finding vector First and Second */
                subtractVV(d, pX + d*j, pX + d*i, First);
                subtractVV(d, pX + d*l, pX + d*i, Second);

                /* step size */
                eta = 1/(beta_in*iters);
                
                multMS(d, M, 1.0 - 1.0/iters);
                add_MM(d, M, pM0, 1.0/iters);
                
                multMSV(d, M, Second, eta);
                multMSV(d, M, First, -eta);
                
                /* finding the smallest eigenvalue */
                eta = getSmallestEigenvalue(k,d,M,v,alpha,beta,Q,Z,temp,tempM);
                
                /* if the matrix contains a negative eigenvalue */
                if (eta < 0)
                    multMSV(d, M, v, -eta);
            }else{
                multMS(d, M, 1.0 - 1.0/iters);
                add_MM(d, M, pM0, 1.0/iters);
            }
            /* scaling the matrix*/
            /*multMS(d, M, min(1.0, 1.0/sqrt(beta_in*frobenius_norm(d, M))));*/
        }
        
        cost = getCostFunct(d, pX, m, pT, pM0, beta_in, M);        
        if (pre_cost - cost < 1e-6)
            break;
        
        pre_cost = cost;
        
        if (DEBUG)            
            mexPrintf("#epoch=%d, Cost=%.10f\n", epoch, cost);
    }
    
    /* release memory */
    free(temp);
    free(First);
    free(Second);
    free(order);
    free(alpha);
    free(beta);
    free(v);
    destroyRM(Z, k + 1);
    destroyRM(Q, d);
    destroyRM(tempM, d);
    
    return cost;
}


void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[] )
{

    R_VEC pX, pM0, pM;
    I_VEC pT;
    double beta;
    int max_iters;
    
    R_MAT M;
    
    double c, lr;
    int d, m;
    int i, j;
    
    if (nrhs != EXACTINPUTS) {
        mexPrintf("Check input parameters\n");    
        return;
    }
    
    /* get inputs */
    pX     = (R_VEC) mxGetPr(INPUT_X);
    pT     = (I_VEC) mxGetPr(INPUT_T);
    pM0    = (R_VEC) mxGetPr(INPUT_M0);
    beta   = mxGetScalar(INPUT_beta);
    max_iters = (int) mxGetScalar(INPUT_max_iters);
    
    d      = mxGetM(INPUT_X);
    m      = mxGetN(INPUT_T);    
    
    /* build output configure */
    OUTPUT_C = mxCreateDoubleMatrix(1, 1, mxREAL);
    OUTPUT_M = mxCreateDoubleMatrix(d, d, mxREAL);
    pM       = (R_VEC)mxGetPr(OUTPUT_M);
    
    /* make dummy matrix for learning*/
    M = createRM(d, d);
    
    *mxGetPr(OUTPUT_C) = distanceMetricLearning(d, pX, m, pT, pM0, beta, M, max_iters); 
     
                             
    for (i = 0; i < d; ++ i)
        for(j = 0; j < d; ++ j)
            pM[i*d+j] = M[i][j];
    
    /* free memory */
    destroyRM(M, d);
}







