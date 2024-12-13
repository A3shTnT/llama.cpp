#include "common.cuh"
#include "mmv.cuh"

template <typename type_acc, int64_t template_cols2>
static __global__ void mul_mat_vec(const half * __restrict__ x, const float * __restrict__ y, float * __restrict__ dst,
                                   const int64_t nrows, const int64_t ncols2, const int64_t stride_row,
                                   const int64_t channel_ratio, const int64_t stride_channel_x,
                                   const int64_t stride_channel_y, const int64_t stride_channel_dst) {
    const int64_t n_cols2   = template_cols2 == 0 ? ncols2 : template_cols2;
    const int     block_row = 4;

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const int tid  = threadIdx.x;
    const int wid  = tid / WARP_SIZE;
    const int wtid = tid % WARP_SIZE;

    const half *  block_x   = x + (bidx / channel_ratio) * stride_channel_x + bidy * block_row * stride_row;
    const float * block_y   = y + bidx * stride_channel_y;
    float *       block_dst = dst + bidx * stride_channel_dst + bidy * block_row;

    const half2 *  block_x2 = (const half2 *) block_x;
    const float2 * block_y2 = (const float2 *) block_y;

    float sumf = 0.0f;

#pragma unroll
    for (int64_t i = wtid; i < n_cols2; i += WARP_SIZE) {
        float2 tmpx = { 0.0f, 0.0f };
        float2 tmpy = { 0.0f, 0.0f };
        if (i < n_cols2 && bidy * block_row + wid < nrows) {
            tmpx = __half22float2(block_x2[wid * stride_row / 2 + i]);
            tmpy = block_y2[i];
        }

        sumf += tmpx.x * tmpy.x;
        sumf += tmpx.y * tmpy.y;
    }

    sumf = warp_reduce_sum(sumf);

    if (wtid == 0 && bidy * block_row + wid < nrows) {
        block_dst[wid] = sumf;
    }
}

template <typename type_acc>
static void launch_mul_mat_vec_cuda(const half * x, const float * y, float * dst, const int64_t ncols,
                                    const int64_t nrows, const int64_t stride_row, const int64_t nchannels_x,
                                    const int64_t nchannels_y, const int64_t stride_channel_x,
                                    const int64_t stride_channel_y, const int64_t stride_channel_dst,
                                    cudaStream_t stream) {
    GGML_ASSERT(ncols % 2 == 0);
    GGML_ASSERT(stride_row % 2 == 0);
    GGML_ASSERT(nchannels_y % nchannels_x == 0);

    const int64_t channel_ratio = nchannels_y / nchannels_x;

    const int block_row = 4;

    int64_t block_size = 128;

    const dim3 block_nums(nchannels_y, (nrows + block_row - 1) / block_row, 1);
    const dim3 block_dims(block_size, 1, 1);

    switch (ncols) {
        case 32:
            {
                mul_mat_vec<type_acc, 16>
                    <<<block_nums, block_dims, 0, stream>>>(x, y, dst, nrows, ncols / 2, stride_row, channel_ratio,
                                                            stride_channel_x, stride_channel_y, stride_channel_dst);
            }
            break;
        case 64:
            {
                mul_mat_vec<type_acc, 32>
                    <<<block_nums, block_dims, 0, stream>>>(x, y, dst, nrows, ncols / 2, stride_row, channel_ratio,
                                                            stride_channel_x, stride_channel_y, stride_channel_dst);
            }
            break;
        case 128:
            {
                mul_mat_vec<type_acc, 64>
                    <<<block_nums, block_dims, 0, stream>>>(x, y, dst, nrows, ncols / 2, stride_row, channel_ratio,
                                                            stride_channel_x, stride_channel_y, stride_channel_dst);
            }
            break;
        case 256:
            {
                mul_mat_vec<type_acc, 128>
                    <<<block_nums, block_dims, 0, stream>>>(x, y, dst, nrows, ncols / 2, stride_row, channel_ratio,
                                                            stride_channel_x, stride_channel_y, stride_channel_dst);
            }
            break;
        case 512:
            {
                mul_mat_vec<type_acc, 256>
                    <<<block_nums, block_dims, 0, stream>>>(x, y, dst, nrows, ncols / 2, stride_row, channel_ratio,
                                                            stride_channel_x, stride_channel_y, stride_channel_dst);
            }
            break;
        case 1024:
            {
                mul_mat_vec<type_acc, 512>
                    <<<block_nums, block_dims, 0, stream>>>(x, y, dst, nrows, ncols / 2, stride_row, channel_ratio,
                                                            stride_channel_x, stride_channel_y, stride_channel_dst);
            };
            break;
        case 2048:
            {
                mul_mat_vec<type_acc, 1024>
                    <<<block_nums, block_dims, 0, stream>>>(x, y, dst, nrows, ncols / 2, stride_row, channel_ratio,
                                                            stride_channel_x, stride_channel_y, stride_channel_dst);
            }
            break;
        case 4096:
            {
                mul_mat_vec<type_acc, 2048>
                    <<<block_nums, block_dims, 0, stream>>>(x, y, dst, nrows, ncols / 2, stride_row, channel_ratio,
                                                            stride_channel_x, stride_channel_y, stride_channel_dst);
            }
            break;
        default:
            {
                mul_mat_vec<type_acc, 0><<<block_nums, block_dims, 0, stream>>>(x, y, dst, nrows, ncols / 2, stride_row,
                                                                                channel_ratio, stride_channel_x,
                                                                                stride_channel_y, stride_channel_dst);
            }
            break;
    }
}

static void mul_mat_vec_cuda(const half * x, const float * y, float * dst, const int64_t ncols, const int64_t nrows,
                             const int64_t stride_row, const int64_t nchannels_x, const int64_t nchannels_y,
                             const int64_t stride_channel_x, const int64_t stride_channel_y,
                             const int64_t stride_channel_dst, enum ggml_prec prec, cudaStream_t stream) {
    switch (prec) {
        case GGML_PREC_DEFAULT:
            {
                launch_mul_mat_vec_cuda<half>(x, y, dst, ncols, nrows, stride_row, nchannels_x, nchannels_y,
                                              stride_channel_x, stride_channel_y, stride_channel_dst, stream);
            }
            break;
        case GGML_PREC_F32:
            {
                launch_mul_mat_vec_cuda<float>(x, y, dst, ncols, nrows, stride_row, nchannels_x, nchannels_y,
                                               stride_channel_x, stride_channel_y, stride_channel_dst, stream);
            }
            break;
    }
}

void ggml_cuda_mul_mat_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                           ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];

    GGML_ASSERT(src1->ne[1] == 1);

    const int            cc   = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    const half *  src0_d = (const half *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) dst->data;

    const int64_t ne02 = src0->ne[2];
    const int64_t ne12 = src1->ne[2];
    GGML_ASSERT(dst->ne[2] == ne12);

    GGML_ASSERT(src0->ne[3] == 1);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(dst->ne[3] == 1);

    const int64_t stride_row         = src0->nb[1] / ggml_type_size(src0->type);
    const int64_t channel_stride_x   = src0->nb[2] / ggml_type_size(src0->type);
    const int64_t channel_stride_y   = src1->nb[2] / ggml_type_size(src1->type);
    const int64_t channel_stride_dst = dst->nb[2] / ggml_type_size(dst->type);

    mul_mat_vec_cuda(src0_d, src1_d, dst_d, ne00, ne01, stride_row, ne02, ne12, channel_stride_x, channel_stride_y,
                     channel_stride_dst, prec, ctx.stream());
}

void ggml_cuda_op_mul_mat_vec(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1,
                              ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
                              const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high,
                              const int64_t src1_ncols, const int64_t src1_padded_row_size, cudaStream_t stream) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t ne00     = src0->ne[0];
    const int64_t row_diff = row_high - row_low;

    GGML_ASSERT(src1_ncols == 1);

    const int            cc   = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const enum ggml_prec prec = fast_fp16_available(cc) ? ggml_prec(dst->op_params[0]) : GGML_PREC_F32;

    // ggml_cuda_op provides single, contiguous matrices
    const int64_t stride_row         = ne00;
    const int64_t nchannels_x        = 1;
    const int64_t nchannels_y        = 1;
    const int64_t channel_stride_x   = 0;
    const int64_t channel_stride_y   = 0;
    const int64_t channel_stride_dst = 0;

    mul_mat_vec_cuda((const half *) src0_dd_i, src1_ddf_i, dst_dd_i, ne00, row_diff, stride_row, nchannels_x,
                     nchannels_y, channel_stride_x, channel_stride_y, channel_stride_dst, prec, stream);

    GGML_UNUSED(ctx);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
}
