#include <THC/THC.h>

#include "ellipsoid_query_gpu.h"

extern THCState *state;

int ellipsoid_query_wrapper(int b, int n, int m, float e1, float e2, float e3, int nsample,
		       THCudaTensor *new_xyz_tensor, THCudaTensor *xyz_tensor, THCudaIntTensor *fps_idx_tensor,
		       THCudaIntTensor *idx_tensor,THCudaIntTensor  *ingroup_pts_cnt_tensor, THCudaTensor *ingroup_out_tensor, THCudaTensor *ingroup_cva_tensor, THCudaTensor *v_tensor,THCudaTensor *d_tensor) {

    const float *new_xyz = THCudaTensor_data(state, new_xyz_tensor);
    const float *xyz = THCudaTensor_data(state, xyz_tensor);
    const int *fps_idx = THCudaIntTensor_data(state, fps_idx_tensor);
    int *idx = THCudaIntTensor_data(state, idx_tensor);
    //below tensors added by me
    int *ingroup_pts_cnt = THCudaIntTensor_data(state, ingroup_pts_cnt_tensor);
    float *ingroup_out = THCudaTensor_data(state, ingroup_out_tensor);
    float *ingroup_cva = THCudaTensor_data(state, ingroup_cva_tensor);
    float *v = THCudaTensor_data(state, v_tensor);
    float *d = THCudaTensor_data(state, d_tensor);

    cudaStream_t stream = THCState_getCurrentStream(state);

    query_ellipsoid_point_kernel_wrapper(b, n, m, e1, e2, e3, nsample, new_xyz, xyz, fps_idx, idx, ingroup_pts_cnt, ingroup_out, ingroup_cva, v, d,
				    stream);
    return 1;
}
