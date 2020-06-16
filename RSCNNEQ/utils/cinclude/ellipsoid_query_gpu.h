#ifndef _ELLIPSOID_QUERY_GPU
#define _ELLIPSOID_QUERY_GPU

#ifdef __cplusplus
extern "C" {
#endif

void query_ellipsoid_point_kernel_wrapper(int b, int n, int m, float e1, float e2, float e3,
				     int nsample, const float *xyz,
				     const float *new_xyz, const int *fps_idx, int *idx, int *ingroup_pts_cnt, 
					 float *ingroup_out, float *ingroup_cva, float *v, float *d,
				     cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
