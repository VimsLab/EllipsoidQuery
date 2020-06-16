
int ellipsoid_query_wrapper(int b, int n, int m, float e1, float e2, float e3, int nsample,
		       THCudaTensor *new_xyz_tensor, THCudaTensor *xyz_tensor, THCudaIntTensor *fps_idx_tensor,
		       THCudaIntTensor *idx_tensor, THCudaIntTensor  *ingroup_pts_cnt_tensor, 
			   THCudaTensor *ingroup_out_tensor, THCudaTensor *ingroup_cva_tensor, 
			   THCudaTensor *v_tensor, THCudaTensor *d_tensor);