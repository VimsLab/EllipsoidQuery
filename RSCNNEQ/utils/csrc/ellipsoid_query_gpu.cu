#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <omp.h>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include <vector>
#include <algorithm>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <cstdlib>
#include <device_launch_parameters.h>
#include "helper_cuda.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <unistd.h>
#include <sys/time.h>
#include <map>
#include <assert.h>
#include "ellipsoid_query_gpu.h"
#include "cuda_utils.h"
int row = 0;
int col = 0;
using namespace std;
const int max_iter = 1000;

/* ---------------------------------------------------------------- */
//
// the following functions come from here:
//
// https://people.sc.fsu.edu/~jburkardt/cpp_src/jacobi_eigenvalue/jacobi_eigenvalue.cpp
//
// attributed to j. burkardt, FSU
// they are unmodified except to add __host__ __device__ decorations
//
//****************************************************************************80
__device__ void r8mat_diag_get_vector(int n, float a[], float v[])
{
  int i;

  for ( i = 0; i < n; i++ )
  {
    v[i] = a[i+i*n];
  }

  return;
}
//****************************************************************************80
__device__ void r8mat_identity(int n, float a[])
{
  int i;
  int j;
  int k;

  k = 0;
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < n; i++ )
    {
      if ( i == j )
      {
        a[k] = 1.0;
      }
      else
      {
        a[k] = 0.0;
      }
      k = k + 1;
    }
  }

  return;
}
//****************************************************************************80
__device__ void jacobi_eigenvalue(int n, float a[], int it_max, float v[], float d[], int &it_num, int &rot_num)
{
  float *bw;
  float c;
  float g;
  float gapq;
  float h;
  int i;
  int j;
  int k;
  int l;
  int m;
  int p;
  int q;
  float s;
  float t;
  float tau;
  float term;
  float termp;
  float termq;
  float theta;
  float thresh;
  float w;
  float *zw;

  r8mat_identity ( n, v );

  r8mat_diag_get_vector ( n, a, d );

  bw = new float[n];
  zw = new float[n];

  for ( i = 0; i < n; i++ )
  {
    bw[i] = d[i];
    zw[i] = 0.0;
  }
  it_num = 0;
  rot_num = 0;

  while ( it_num < it_max )
  {
    it_num = it_num + 1;
//
//  The convergence threshold is based on the size of the elements in
//  the strict upper triangle of the matrix.
//
    thresh = 0.0;
    for ( j = 0; j < n; j++ )
    {
      for ( i = 0; i < j; i++ )
      {
        thresh = thresh + a[i+j*n] * a[i+j*n];
      }
    }

    thresh = sqrt ( thresh ) / ( float ) ( 4 * n );

    if ( thresh == 0.0 )
    {
      break;
    }

    for ( p = 0; p < n; p++ )
    {
      for ( q = p + 1; q < n; q++ )
      {
        gapq = 10.0 * fabs ( a[p+q*n] );
        termp = gapq + fabs ( d[p] );
        termq = gapq + fabs ( d[q] );
//
//  Annihilate tiny offdiagonal elements.
//
        if ( 4 < it_num &&
             termp == fabs ( d[p] ) &&
             termq == fabs ( d[q] ) )
        {
          a[p+q*n] = 0.0;
        }
//
//  Otherwise, apply a rotation.
//
        else if ( thresh <= fabs ( a[p+q*n] ) )
        {
          h = d[q] - d[p];
          term = fabs ( h ) + gapq;

          if ( term == fabs ( h ) )
          {
            t = a[p+q*n] / h;
          }
          else
          {
            theta = 0.5 * h / a[p+q*n];
            t = 1.0 / ( fabs ( theta ) + sqrt ( 1.0 + theta * theta ) );
            if ( theta < 0.0 )
            {
              t = - t;
            }
          }
          c = 1.0 / sqrt ( 1.0 + t * t );
          s = t * c;
          tau = s / ( 1.0 + c );
          h = t * a[p+q*n];
//
//  Accumulate corrections to diagonal elements.
//
          zw[p] = zw[p] - h;                 
          zw[q] = zw[q] + h;
          d[p] = d[p] - h;
          d[q] = d[q] + h;

          a[p+q*n] = 0.0;
//
//  Rotate, using information from the upper triangle of A only.
//
          for ( j = 0; j < p; j++ )
          {
            g = a[j+p*n];
            h = a[j+q*n];
            a[j+p*n] = g - s * ( h + g * tau );
            a[j+q*n] = h + s * ( g - h * tau );
          }

          for ( j = p + 1; j < q; j++ )
          {
            g = a[p+j*n];
            h = a[j+q*n];
            a[p+j*n] = g - s * ( h + g * tau );
            a[j+q*n] = h + s * ( g - h * tau );
          }

          for ( j = q + 1; j < n; j++ )
          {
            g = a[p+j*n];
            h = a[q+j*n];
            a[p+j*n] = g - s * ( h + g * tau );
            a[q+j*n] = h + s * ( g - h * tau );
          }
//
//  Accumulate information in the eigenvector matrix.
//
          for ( j = 0; j < n; j++ )
          {
            g = v[j+p*n];
            h = v[j+q*n];
            v[j+p*n] = g - s * ( h + g * tau );
            v[j+q*n] = h + s * ( g - h * tau );
          }
          rot_num = rot_num + 1;
        }
      }
    }

    for ( i = 0; i < n; i++ )
    {
      bw[i] = bw[i] + zw[i];
      d[i] = bw[i];
      zw[i] = 0.0;
    }
  }
//
//  Restore upper triangle of input matrix.
//
  for ( j = 0; j < n; j++ )
  {
    for ( i = 0; i < j; i++ )
    {
      a[i+j*n] = a[j+i*n];
    }
  }
//
//  Ascending sort the eigenvalues and eigenvectors.
//
  for ( k = 0; k < n - 1; k++ )
  {
    m = k;
    for ( l = k + 1; l < n; l++ )
    {
      if ( d[l] < d[m] )
      {
        m = l;
      }
    }

    if ( m != k )
    {
      t    = d[m];
      d[m] = d[k];
      d[k] = t;
      for ( i = 0; i < n; i++ )
      {
        w        = v[i+m*n];
        v[i+m*n] = v[i+k*n];
        v[i+k*n] = w;
      }
    }
  }

  delete [] bw;
  delete [] zw;

  return;
}

void initialize_matrix(int mat_id, int n, float *mat, float *v){

  for (int i = 0; i < n*n; i++) *(v+(mat_id*n*n)+i) = mat[i];
}

// end of FSU code
/* ---------------------------------------------------------------- */

//Ellipsoid querying
// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ellipsoid_point_kernel(int b, int n, int m, float e1, float e2, float e3,
					int nsample,
					const float *__restrict__ new_xyz,
					const float *__restrict__ xyz,
          const int *__restrict__ fps_idx,
          int *__restrict__ idx,
          int *__restrict__ ingroup_pts_cnt,
          float *__restrict__ ingroup_out,
          float *__restrict__ ingroup_cva, 
          float *__restrict__ v, 
          float *__restrict__ d){
    int batch_index = blockIdx.x;
    int c = 3;
    xyz += batch_index * n * 3;
    new_xyz += batch_index * m * 3;
    fps_idx += batch_index * m;
    idx += m * nsample * batch_index;
    ingroup_pts_cnt += m*batch_index;
    ingroup_out += m*nsample*3*batch_index;
    ingroup_cva += m*3*3*batch_index;
    v += m*3*3*batch_index;
    d += m*3*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    //squares of axis-lengths
    float aa = e1 * e1;
    float bb = e2 * e2;
    float cc = e3 * e3;
    for (int j = index; j < m; j += stride) {
      float new_x = new_xyz[j * 3 + 0];
      float new_y = new_xyz[j * 3 + 1];
      float new_z = new_xyz[j * 3 + 2];
      for (int l = 0; l < nsample; ++l) {
          idx[j * nsample + l] = fps_idx[j];
      }
      int cnt = 0;
      for (int k = 0; k < n && cnt < nsample; ++k) {
          float x = xyz[k * 3 + 0];
          float y = xyz[k * 3 + 1];
          float z = xyz[k * 3 + 2];
          //first round of ellipsoid querying
          float d2 = max(sqrtf(((new_x - x) * (new_x - x)/aa) + ((new_y - y) * (new_y - y)/bb) +
              ((new_z - z) * (new_z - z)/cc)),1e-20f);
          if (d2 <= 1 && d2 > 0) {
            idx[j * nsample + cnt] = k;
            ++cnt;
          }
      }
      ingroup_pts_cnt[j] = cnt;

      //grouping of ellipsoid-queried points
      for (int k=0;k<nsample;++k) {
        int ii = idx[j*nsample+k];
        for (int l=0;l<c;++l) {
            ingroup_out[j*nsample*c+k*c+l] = xyz[ii*c+l];
        }
      }
      //from the grouped points pick unique points
      float *Matrix=(float *)malloc(sizeof(float)*ingroup_pts_cnt[j]*c);
      float *tMatrix=(float *)malloc(sizeof(float)*ingroup_pts_cnt[j]*c);
      int flag=0;
      if(ingroup_pts_cnt[j]>=3){
        for(int k=0;k<ingroup_pts_cnt[j];k++){
          int ii = idx[j*nsample+k];
          Matrix[0+3*k] = xyz[ii*c+0];
          Matrix[1+3*k] = xyz[ii*c+1];
          Matrix[2+3*k] = xyz[ii*c+2];
          if(xyz[ii*c+0]==0 && xyz[ii*c+1]==0 && xyz[ii*c+2]==0){
            flag=1;
          }
        }
        if(flag!=1){      
          //find mean of unique points
          float means[3];
          float d2;
          means[0]=means[1]=means[2]=0.0;               
          for (int up=0;up<ingroup_pts_cnt[j];up++){
            means[0]+=Matrix[up*c+0];
            means[1]+=Matrix[up*c+1];
            means[2]+=Matrix[up*c+2];                   
          }
          means[0]=means[0]/ingroup_pts_cnt[j];
          means[1]=means[1]/ingroup_pts_cnt[j];
          means[2]=means[2]/ingroup_pts_cnt[j];

          //distance between mean of unique points and the centroid point
          d2=sqrtf((means[0]-new_x)*(means[0]-new_x)+(means[1]-new_y)*(means[1]-new_y)+(means[2]-new_z)*(means[2]-new_z));

          //covariance adjustment
          if (d2 >= e1/4.0){
            //if more points are on one side of the centroid
            for(int up=0;up<ingroup_pts_cnt[j];up++){
              //subtract centroid from the points
              Matrix[c*up]=Matrix[c*up]-new_x;
              Matrix[c*up+1]=Matrix[c*up+1]-new_y;
              Matrix[c*up+2]=Matrix[c*up+2]-new_z;
            }
          }else{
            for(int up=0;up<ingroup_pts_cnt[j];up++){
              // subtract mean from the points
              Matrix[c*up]=Matrix[c*up]-means[0];
              Matrix[c*up+1]=Matrix[c*up+1]-means[1];
              Matrix[c*up+2]=Matrix[c*up+2]-means[2];
            }
          }
          //transpose points matrix
          for(int tpt=0;tpt<c;tpt++){
            for(int tup=0;tup<ingroup_pts_cnt[j];tup++){
              tMatrix[tpt+c*tup]=Matrix[tpt+c*tup];
            }
          }
          //calculate covariance matrix
          float *covm=(float *)malloc(sizeof(float)*c*c);   
          for(int t3=0;t3<c;t3++){
            for(int tn=0;tn<c;tn++){
              covm[tn+t3*c] = 0.0;
              for(int n3=0;n3<ingroup_pts_cnt[j];n3++){
                covm[tn+t3*c]+=tMatrix[t3+c*n3]*Matrix[tn+n3*c];
              }
              ingroup_cva[j*c*c+tn+t3*c]=covm[tn+t3*c]/(ingroup_pts_cnt[j]-1);
            }
          }
          free(covm);
        }
      }
      free(Matrix);
      free(tMatrix);

      int it_num;
      int rot_num;
      if((ingroup_pts_cnt[j]>=3)){
        //Eigendecomposition 
        jacobi_eigenvalue(c, ingroup_cva+(j*c*c), max_iter, v+(j*c*c), d+(j*c), it_num, rot_num);
        cnt = ingroup_pts_cnt[j];
        for (int k=0;k<n;++k) {
          if (cnt == nsample)
            break; // only pick the FIRST nsample points in the ellipsoid
          float x1=xyz[k*3+0];
          float y1=xyz[k*3+1];
          float z1=xyz[k*3+2];
          float spoint[3];
          float rspoint[3];
          spoint[0]=x1-new_x;
          spoint[1]=y1-new_y;
          spoint[2]=z1-new_z;
          //rotating input points
          rspoint[0] = ((*(v+(c*c*j)+6)))*spoint[0]+((*(v+(c*c*j)+7)))*spoint[1]+((*(v+(c*c*j)+8)))*spoint[2];
          rspoint[1] = ((*(v+(c*c*j)+3)))*spoint[0]+((*(v+(c*c*j)+4)))*spoint[1]+((*(v+(c*c*j)+5)))*spoint[2];
          rspoint[2] = ((*(v+(c*c*j)+0)))*spoint[0]+((*(v+(c*c*j)+1)))*spoint[1]+((*(v+(c*c*j)+2)))*spoint[2];
          float xx = rspoint[0];
          float yy = rspoint[1];
          float zz = rspoint[2];
          //second querying - reoriented ellipsoid
          float d3=max(sqrtf((xx*xx/aa)+(yy*yy/bb)+(zz*zz/cc)),1e-20f);
          //union of both query points
          if (d3<=1) {
            int kflag=0;
            for(int kk=0;kk<nsample;kk++){
              if (idx[j*nsample+kk]==k){
                kflag=1;
                break;
              }
            }
            if (kflag!=1){
              idx[j*nsample+cnt] = k;
              cnt+=1;
            }
          }       
        }
        ingroup_pts_cnt[j] = cnt;
      }
    }
}

void query_ellipsoid_point_kernel_wrapper(int b, int n, int m, float e1, float e2, float e3,
				     int nsample, const float *new_xyz,
             const float *xyz, const int *fps_idx, int *idx, 
             int *ingroup_pts_cnt, float *ingroup_out, 
             float *ingroup_cva, float *v, float *d,
				     cudaStream_t stream) {

    cudaError_t err;
    query_ellipsoid_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, e1, e2, e3, nsample, new_xyz, xyz, fps_idx, idx, ingroup_pts_cnt, ingroup_out, ingroup_cva, v, d);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
	fprintf(stderr, "CUDA kernel failed inside ellipsoid wrapper: %s\n", cudaGetErrorString(err));
	exit(-1);
    }
}
