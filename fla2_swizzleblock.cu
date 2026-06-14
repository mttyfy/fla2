//////// smem 128kb 

#include "flash_attn_kernel.cuh"
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "utils.h"
#include <cuda_fp16.h>
#include <cfloat>
#define div_ceil(a, b) (((a) + (b) - 1) / (b))


template<
const int head_dim,
const int mma_m,
const int mma_n,
const int mma_k,
const int mma_tile_q,
const int mma_tile_k,
const int mma_tile_p,
const int mma_tile_v,
const int mma_warp_q,
const int mma_warp_k,
const int mma_warp_p,
const int mma_warp_v_dim,
const int storagefloate32,
const int stage,
const int padq,
const int padk,
const int padv>
__global__ void __launch_bounds__ (WARP_SIZE * mma_tile_q * mma_tile_k)
    flash_attn_mma_stages_split_q_shared_kv_kernel(half * Q, half * K, half * V, half * O,
        const int QKV_batch, const int head, const int seqlen) {
        if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0)
            printf("Kernel is running!\n");
        const int br = mma_m * mma_tile_q * mma_warp_q;
        const int bc = mma_n * mma_tile_k * mma_warp_k;
        const int numthreads = WARP_SIZE * mma_tile_q * mma_tile_k;
        const int numberof_k_iters = div_ceil(seqlen,bc); 
        const float scale = 1.0f / sqrt((float)head_dim);
        ////////////// id grid
        int batch_id = blockIdx.y / head;
        int head_id = blockIdx.y % head;
        int Q_br_id = blockIdx.x;
        int O_br_id = Q_br_id;
        if (batch_id >= QKV_batch || head_id >= head)
            return;
        int Q_gmem_offset = batch_id * head * seqlen * head_dim + head_id * seqlen * head_dim;
        int K_gmem_offset = Q_gmem_offset;
        int V_gmem_offset = Q_gmem_offset;
        int O_gmem_offset = Q_gmem_offset;
        int tid = threadIdx.x;
        int warp_id = tid / WARP_SIZE;
        int lane_id = tid % WARP_SIZE;
        int warp_qp = warp_id;
        int warp_kv = 0;
        /// smem
        int Q_smem_br = tid / (numthreads / br);
        int Q_smem_k  = tid % (numthreads / br) *((br * head_dim) / numthreads);
        int K_smem_bc = tid / (numthreads / bc);
        int K_smem_k  = tid % (numthreads / bc) *((bc * head_dim) / numthreads);
        int V_smem_bc = tid / (numthreads / bc);
        int V_smem_k  = tid % (numthreads / bc) *((bc * head_dim) / numthreads);
        int Q_gmem_br =  Q_br_id * br + Q_smem_br;
        if (Q_gmem_br >= seqlen)
            return;
        int K_gmem_br = 0;
        int V_gmem_br = 0;
        const int Q_tile_size = br * (head_dim + padq);
        const int K_tile_size = bc * (head_dim + padk);
        const int V_tile_size = bc * (head_dim + padv);
        extern __shared__ half smem[];
        half *Q_smem_tile_start = smem;
        half *K_smem_tile_start = smem + Q_tile_size;
        half *V_smem_tile_start = K_smem_tile_start;
        uint32_t Q_smem_tile_start_ptr = __cvta_generic_to_shared(Q_smem_tile_start);
        uint32_t K_smem_tile_start_ptr = __cvta_generic_to_shared(K_smem_tile_start);
        uint32_t V_smem_tile_start_ptr = __cvta_generic_to_shared(V_smem_tile_start);
        constexpr bool canprefetchQs2r = head_dim < 64;
        constexpr bool candelayprefetchQs2r = canprefetchQs2r;
        constexpr int prefetchQs2rnum = canprefetchQs2r? 8 : 1; 
        constexpr bool canprefetchKVg2s = (stage == 2);
        int prefetchKg2ssmemid = 0;
        int prefetchVg2ssmemid = canprefetchKVg2s? 1 : 0;
        uint32_t RQ[prefetchQs2rnum][mma_warp_q][4];
        ////  "splitdim  splitk sharekv "
        uint32_t RK[mma_warp_k][2];
        uint32_t RV[mma_warp_v_dim][2];
        uint32_t RS[mma_warp_q][mma_warp_k][2];
        uint32_t RO[mma_warp_p][mma_warp_v_dim][2];
        /// softmax
        uint32_t RD[mma_warp_p][mma_warp_v_dim][2];
        fill_3D_regs<uint32_t>(RD,0);
        {
            int Q_gmem_addr = Q_gmem_offset + Q_gmem_br * head_dim + Q_smem_k;
            ///uint32_t Q_smem_addr =Q_smem_tile_start_ptr + 
            ////((Q_smem_br * (head_dim + padq )+ Q_smem_k) * sizeof(half));
            ///// Q_smem[br][]
        
            #pragma unroll
            for (int i = 0 ;  i < (head_dim * br) / numthreads ; i+=8 ) {
                /// "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       
                /// "l"(src), "n"(bytes))
                uint32_t Q_smem_addr =
                Q_smem_tile_start_ptr +
                (
                (((Q_smem_k + i)/ mma_k)  * br * (mma_k + padq ) +
                Q_smem_br * (mma_k + padq )+ 
                swizzle_with_block<mma_k>((Q_smem_k+i)/mma_k,Q_smem_br,
                (Q_smem_k + i) % mma_k)) 
                * sizeof(half));
                if (i==0&&Q_br_id==0&&batch_id==0&&head_id==0) {
                    uint32_t tmp_bank_id = (Q_smem_addr % 128) / 4;
                    uint32_t tmp_bank_row_id = Q_smem_addr / 128;
                  printf(
                        "tid=%d lane=%d i=%d "
                        "Q_smem_br=%d Q_smem_k=%d "
                       "Q_smem_addr=%u 0x%x "  
                       "bank_id=%u " "bank_row_id=%u \n",
                        tid, lane_id, i,
                        Q_smem_br, Q_smem_k,
                        Q_smem_addr,Q_smem_addr
                        ,tmp_bank_id,tmp_bank_row_id
                    );
                }
                CP_ASYNC_CG(Q_smem_addr , &Q[Q_gmem_addr + i] , 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }
        float lane_block_row_max_old[mma_tile_q][2]; // [1][2]
        float lane_block_row_sum_old[mma_tile_q][2]; // [1][2]
        fill_2D_regs<float, mma_tile_q, 2>(lane_block_row_max_old, -FLT_MAX);
        fill_2D_regs<float, mma_tile_q, 2>(lane_block_row_sum_old, 0.0f);
        /// begin K iters
        #pragma unroll
        for (int i = 0 ; i < seqlen / bc ; i += 1) {
            if (canprefetchKVg2s) {
                if (i == 1) {
                    
                    int K_gmem_addr = K_gmem_offset + (i * bc + K_smem_bc) * head_dim 
                    + K_smem_k;
                  
                    #pragma unroll
                    for (int j = 0 ; j <(bc * head_dim) / numthreads ; j+=8) {
                        uint32_t K_smem_addr = 
                        K_smem_tile_start_ptr
                        + (K_tile_size * prefetchKg2ssmemid
                        + ((K_smem_k + j)/mma_k * bc * (mma_k + padk) )
                        + K_smem_bc * (mma_k + padk) + 
                        swizzle<mma_k>(K_smem_bc,((K_smem_k+j)%mma_k))) * sizeof(half);
                        CP_ASYNC_CG(K_smem_addr, &K[K_gmem_addr + j], 16);
                    }
                    CP_ASYNC_COMMIT_GROUP();
                    CP_ASYNC_WAIT_GROUP(0);
                    __syncthreads();
                }
                {
                    int V_gmem_addr = V_gmem_offset + 
                    (i * bc + V_smem_bc) * head_dim + V_smem_k;
                    int V_smem_addr = V_smem_tile_start_ptr + 
                    (prefetchVg2ssmemid * V_tile_size + V_smem_bc * (head_dim + padv) + 
                    V_smem_k) * sizeof(half);
                    #pragma unroll
                    for (int j = 0 ; j < ((bc*head_dim)/numthreads) ; j+=8){
                        uint32_t V_smem_addr = 
                        V_smem_tile_start_ptr
                        + (V_tile_size * prefetchVg2ssmemid
                        + ((V_smem_bc + j)/mma_k * head_dim * (mma_k + padv) )
                        + V_smem_k * (mma_k + padv) + 
                        swizzle<mma_k>(V_smem_k,((V_smem_bc+j)%mma_k))) * sizeof(half);
                        CP_ASYNC_CG(V_smem_addr,&V[V_gmem_addr+j],16);
                    }
                    CP_ASYNC_COMMIT_GROUP();
                }
            } else {
                int K_gmem_addr = K_gmem_offset + (i * bc + K_smem_bc) * head_dim
                 + K_smem_k;
                #pragma unroll
                for (int j = 0 ; j < ((bc*head_dim)/numthreads); j+=8){
                       uint32_t K_smem_addr = 
                        K_smem_tile_start_ptr
                        + (K_tile_size * prefetchKg2ssmemid
                        + ((K_smem_k + j)/mma_k * bc * (mma_k + padk) )
                        + K_smem_bc * (mma_k + padk) + 
                        swizzle<mma_k>(K_smem_bc,((K_smem_k+j)%mma_k))) * sizeof(half);
                    CP_ASYNC_CG(K_smem_addr , &K[K_gmem_addr+j],16);
                }
                CP_ASYNC_COMMIT_GROUP();
                CP_ASYNC_WAIT_GROUP(0);
                __syncthreads();
            }
            
            if constexpr (canprefetchQs2r && !candelayprefetchQs2r ) {
                if (i == 0) {
                    if constexpr (!canprefetchKVg2s) {
                        CP_ASYNC_WAIT_GROUP(0);
                    } else {
                        CP_ASYNC_WAIT_GROUP(1);
                    }
                    __syncthreads();
                    #pragma unroll
                    for (int k = 0 ; k < head_dim / mma_k ; k ++) {
                        #pragma unroll
                        for (int j = 0 ; j < mma_warp_q ; j++) {
                            int warp_smem_br = warp_id * mma_warp_q *mma_m + j * mma_m;
                            int lane_smem_Q_br = warp_smem_br + lane_id % 16;
                            int lane_smem_Q_k  =  (lane_id / 16  ) * 8;
                            uint32_t lane_smem_addr = 
                            Q_smem_tile_start_ptr +
                            (k * br * mma_k + lane_smem_Q_br * mma_k
                            + swizzle_with_block<mma_k>
                            (k,lane_smem_Q_br,lane_smem_Q_k)) 
                            * sizeof(half);
                            LDMATRIX_X4(RQ[k][j][0],RQ[k][j][1],RQ[k][j][2],
                            RQ[k][j][3],lane_smem_addr);
                        }
                    }
                    __syncthreads();
                }
            }
            fill_3D_regs<uint32_t>(RS,0);
            #pragma unroll
            for (int k = 0 ; k < head_dim/mma_k;k++) {
                if  (!canprefetchQs2r) {
                    #pragma unroll
                    for (int j = 0 ; j < mma_warp_q ; j++) {
                        int warp_smem_Q_br = warp_id * mma_warp_q * mma_m + j * mma_m;
                        int lane_smem_Q_br = warp_smem_Q_br +  lane_id % 16 ;
                        int lane_smem_Q_k =  (lane_id / 16) * 8;
                        uint32_t lane_smem_addr = 
                          Q_smem_tile_start_ptr +
                            (k * br * mma_k + lane_smem_Q_br * mma_k
                            + swizzle_with_block<mma_k>
                            (k,lane_smem_Q_br,lane_smem_Q_k)) 
                            * sizeof(half);
                        LDMATRIX_X4(RQ[0][j][0],RQ[0][j][1],RQ[0][j][2],
                            RQ[0][j][3],lane_smem_addr);
                    }
                } else {
                  if (i == 0 ) {
                    if (k==0) {
                        if constexpr (!canprefetchKVg2s) {
                            CP_ASYNC_WAIT_GROUP(0);
                        } else {
                            CP_ASYNC_WAIT_GROUP(1);
                        }
                        __syncthreads();
                    }
                        #pragma unroll
                        for (int j = 0 ; j < mma_warp_q ; j ++) {
                            int warp_smem_Q_br = warp_id * mma_warp_q * mma_m + j * mma_m;
                            int lane_smem_Q_br = warp_smem_Q_br + lane_id % 16; 
                            int lane_smem_Q_k  = (warp_id / 16 ) * 8;
                            uint32_t lane_smem_addr = 
                              Q_smem_tile_start_ptr +
                            (k * br * mma_k + lane_smem_Q_br * mma_k
                            + swizzle_with_block<mma_k>
                            (k,lane_smem_Q_br,lane_smem_Q_k)) 
                            * sizeof(half);
                            LDMATRIX_X4(RQ[k][j][0],RQ[k][j][1],RQ[k][j][2],
                                RQ[k][j][3],
                            lane_smem_addr);
                        }
                    
                    }
                }

                #pragma unroll
                for (int j = 0 ; j < mma_warp_k; j++) {
                    int WARPK=0;
                    int warp_smem_K_bc = WARPK * mma_warp_k * mma_n + j * mma_n;
                    int lane_smem_K_bc = warp_smem_K_bc + lane_id %8;
                    int lane_smem_K_k  =  ((lane_id / 8) %2 ) * 8;
                    uint32_t lane_smem_addr = K_smem_tile_start_ptr + 
                    (prefetchKg2ssmemid * K_tile_size + 
                    k * bc * (mma_k + padk) + lane_smem_K_bc * (mma_k + padk) 
                    + swizzle<mma_k>(lane_smem_K_bc,lane_smem_K_k) 
                    ) * sizeof(half);
                    LDMATRIX_X2(RK[j][0],RK[j][1],lane_smem_addr);
                }
                if (canprefetchQs2r) {
                    #pragma unroll
                    for (int j = 0 ; j < mma_warp_k ; j++) {
                        HMMA16816(RS[0][j][0],RS[0][j][1],RQ[k][0][0],RQ[k][0][1]
                            ,RQ[k][0][2],RQ[k][0][3],RK[j][0],RK[j][1],
                            RS[0][j][0],RS[0][j][1]);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0 ; j < mma_warp_k ; j++) {
                        HMMA16816(RS[0][j][0],RS[0][j][1],
                        RQ[0][0][0],RQ[0][0][1],RQ[0][0][2],RQ[0][0][3],
                        RK[j][0],RK[j][1],RS[0][j][0],RS[0][j][1]);
                    }
                
            }
        }
        __syncthreads();
        /// bc will slip in seqlen dimension


        if constexpr (!canprefetchKVg2s) {
            int V_gmem_addr = 
            V_gmem_offset + (i * bc + V_smem_bc) * head_dim  + V_smem_k;

            #pragma unroll
            for (int j = 0 ; j < ((bc * head_dim/numthreads)) ; j+=8 ) {
            uint32_t V_smem_addr = V_smem_tile_start_ptr + 
            (
                (V_smem_bc + j) / mma_k * head_dim *  (mma_k + padv) + 
                V_smem_k * (mma_k + padv) + 
                swizzle<mma_k>(V_smem_k,(V_smem_bc+j)%mma_k)
            ) * sizeof(half);
                CP_ASYNC_CG(V_smem_addr , &V[V_gmem_addr + j] , 16);
            }
            CP_ASYNC_COMMIT_GROUP();
        }
        if constexpr (canprefetchKVg2s) {
            if ((i+1) < (seqlen/bc) ) {
                 int K_gmem_addr = K_gmem_offset + ((i+1) * bc + K_smem_bc) * head_dim
                 + K_smem_k;
                #pragma unroll
                for (int j = 0 ; j < ((bc*head_dim)/numthreads); j+=8){
                       uint32_t K_smem_addr = 
                        K_smem_tile_start_ptr
                        + (K_tile_size * prefetchKg2ssmemid
                        + ((K_smem_k + j)/mma_k * bc * (mma_k + padk) )
                        + K_smem_bc * (mma_k + padk) + 
                        swizzle<mma_k>(K_smem_bc,((K_smem_k+j)%mma_k))) * sizeof(half);
                    CP_ASYNC_CG(K_smem_addr, &K[K_gmem_addr+j],16);
                }
            CP_ASYNC_COMMIT_GROUP();
            }
        }

        /////////////////////////online safe soft max ///////////////////////////////////
        ///// template <typename T, const int kWarpSize = WARP_SIZE>
        ///// DEVICE_INLINE T warp_reduce_max(T val) {
        ///// #pragma unroll
        ///// for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        ///// val = max(val, __shfl_xor_sync(0xffffffff, val, mask, kWarpSize));
        ///// }
        ///// return val;
        ///// }
        float lane_row_max_new[mma_warp_q][2];
        float lane_row_sum_new[mma_warp_q][2];
        fill_2D_regs<float,mma_warp_q,2>(lane_row_max_new,-FLT_MAX);
        fill_2D_regs<float,mma_warp_q,2>(lane_row_sum_new,0);
        {
            #pragma unroll
            for (int j = 0 ; j < mma_warp_k ; j++) {
                half *thread_half_s = reinterpret_cast<half *>(&(RS[0][j][0]));
                float lane_max_0 = __half2float(__hmax(thread_half_s[0],thread_half_s[1])) * scale;
                float lane_max_1 = __half2float(__hmax(thread_half_s[2],thread_half_s[3])) * scale;
                lane_row_max_new[0][0] = max(lane_row_max_new[0][0],lane_max_0);
                lane_row_max_new[0][1] = max(lane_row_max_new[0][1],lane_max_1);
            }
            lane_row_max_new[0][0] = warp_reduce_max<float,4>(lane_row_max_new[0][0]);
            lane_row_max_new[0][1] = warp_reduce_max<float,4>(lane_row_max_new[0][1]);
        }
        {
            float block_row_max_new_0 = lane_row_max_new[0][0];
            float block_row_max_new_1 = lane_row_max_new[0][1];
            float block_row_max_old_0 = lane_block_row_max_old[0][0];
            float block_row_max_old_1 = lane_block_row_max_old[0][1];
            block_row_max_new_0 = max(block_row_max_new_0,block_row_max_old_0);
            block_row_max_new_1 = max(block_row_max_new_1,block_row_max_new_1);
            #pragma unroll
            for (int j = 0 ; j < mma_warp_k ; j++) {
                half * thread_half_RS = reinterpret_cast<half *>(&(RS[0][j][0]));
                thread_half_RS[0] = __expf(__fmaf_rn(__half2float(thread_half_RS[0]),scale
                    , -block_row_max_new_0));
                thread_half_RS[1] = __expf(__fmaf_rn(__half2float(thread_half_RS[1]),scale
                    ,-block_row_max_new_0));
                thread_half_RS[2] = __expf(__fmaf_rn(__half2float(thread_half_RS[2]),scale
                    ,-block_row_max_new_1));
                thread_half_RS[3] = __expf(__fmaf_rn(__half2float(thread_half_RS[3]),scale
                    ,-block_row_max_new_1));
                lane_row_sum_new[0][0] += __half2float(thread_half_RS[0]) + __half2float(thread_half_RS[1]);
                lane_row_sum_new[0][1] += __half2float(thread_half_RS[2]) + __half2float(thread_half_RS[3]);
                thread_half_RS[0] = __float2half(thread_half_RS[0]);
                thread_half_RS[1] = __float2half(thread_half_RS[1]);
                thread_half_RS[2] = __float2half(thread_half_RS[2]);
                thread_half_RS[3] = __float2half(thread_half_RS[3]);
            }
            lane_row_sum_new[0][0] = warp_reduce_sum<float,4>(lane_row_sum_new[0][0]);
            lane_row_sum_new[0][1] = warp_reduce_sum<float,4>(lane_row_sum_new[0][1]);
        }
        if (canprefetchKVg2s) {
            if (i + 1 < seqlen / bc ) {
                CP_ASYNC_WAIT_GROUP(1); 
            } else {
                CP_ASYNC_WAIT_GROUP(0); 
            }
        } else {
            CP_ASYNC_WAIT_GROUP(0);
        }
        __syncthreads();
        ///  br * bc        *     bc * head_dim
        /// br * bc has been stored in 4warp 16 * 64 
        fill_3D_regs<uint32_t>(RO, 0);
        #pragma unroll
        for(int k = 0; k < (bc / mma_k) ; k++ ) {
            #pragma unroll
            for (int j = 0 ; j < mma_warp_v_dim ; j++) {
                int lane_smem_V_bc =   k * mma_k + lane_id %16;      
                int lane_smem_V_k  =  warp_kv * mma_warp_v_dim * mma_n + j * mma_n; 
                uint32_t lane_smem_V_addr = V_smem_tile_start_ptr + 
                (
                V_tile_size * prefetchVg2ssmemid + 
                k * head_dim * (mma_k + padv) + 
                lane_smem_V_k * (mma_k + padv) + 
                swizzle<mma_k>(lane_smem_V_k,lane_smem_V_bc)
                ) * sizeof(half);
                LDMATRIX_X2_T(RV[j][0],RV[j][1],lane_smem_V_addr);
            }
            #pragma unroll
            for (int j = 0 ; j < mma_warp_v_dim ; j++) {
                HMMA16816(RO[0][j][0],RO[0][j][1],
                RS[0][2*k][0],RS[0][2*k][1],RS[0][2*k+1][0],RS[0][2*k+1][1],
                RV[j][0],RV[j][1],RO[0][j][0],RO[0][j][1]);
            }
        }

        __syncthreads();
        {
            float block_row_max_new_0 = lane_row_max_new[0][0];
            float block_row_max_new_1 = lane_row_max_new[0][1];
            float block_row_sum_new_0 = lane_row_sum_new[0][0];
            float block_row_sum_new_1 = lane_row_sum_new[0][1];
            float block_row_max_old_0 = lane_block_row_max_old[0][0];
            float block_row_max_old_1 = lane_block_row_max_old[0][1];

            block_row_max_new_0 = max(block_row_max_new_0,block_row_max_old_0);
            block_row_max_new_1 = max(block_row_max_new_1,block_row_max_old_1);
            block_row_max_old_0 = i > 0 ? block_row_max_old_0 : block_row_max_new_0;
            block_row_max_old_1 = i > 0 ? block_row_max_old_1 : block_row_max_new_1;
            float rescale_o_factor_0 = __expf(block_row_max_old_0 - block_row_max_new_0);
            float rescale_o_factor_1 = __expf(block_row_max_old_1 - block_row_max_new_1);

            ///////////////////////////////////////////////////////////////////////////
            ///// 16 * 64
            #pragma unroll
            for(int j = 0 ; j < mma_warp_v_dim ; j++) {
                half * thread_half_RO = reinterpret_cast<half *>(&(RO[0][j][0]));
                half * thread_half_RD = reinterpret_cast<half *>(&(RD[0][j][0]));
                thread_half_RD[0] = __fmaf_rn(rescale_o_factor_0,thread_half_RD[0],
                __half2float(thread_half_RO[0]));
                thread_half_RD[1] = __fmaf_rn(rescale_o_factor_0,thread_half_RD[1],
                __half2float(thread_half_RO[1]));
                thread_half_RD[2] = __fmaf_rn(rescale_o_factor_1,thread_half_RD[2],
                __half2float(thread_half_RO[2]));
                thread_half_RD[3] = __fmaf_rn(rescale_o_factor_1,thread_half_RD[3],
                __half2float(thread_half_RO[3]));
            }
            float block_row_sum_old_0 = lane_block_row_sum_old[0][0];
            float block_row_sum_old_1 = lane_block_row_sum_old[0][1];
            lane_block_row_sum_old[0][0] = (__fmaf_rn(rescale_o_factor_0,block_row_sum_old_0,
            block_row_sum_new_0));
            lane_block_row_sum_old[0][1] = (__fmaf_rn(rescale_o_factor_1,block_row_sum_old_1,
             block_row_sum_new_1));
        }
    }
    
    ////// end for oi * di 
    /////// last /di
    {
        float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[0][0]);
        float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[0][1]);
        /// write back
        #pragma unroll
        for (int j = 0 ;  j < mma_warp_v_dim ; j++) {
            half * thread_half_RD = reinterpret_cast<half *>(&(RD[0][j][0]));
            thread_half_RD[0] = __float2half_rn(rescale_factor_0  * __half2float(thread_half_RD[0]));
            thread_half_RD[1] = __float2half_rn(__half2float(thread_half_RD[1]) * rescale_factor_0);
            thread_half_RD[2] = __float2half_rn(__half2float(thread_half_RD[2]) * rescale_factor_1);
            thread_half_RD[3] = __float2half_rn(__half2float(thread_half_RD[3]) * rescale_factor_1);
        }
    }
    ///// write back
    {
        #pragma unroll
        for (int j = 0 ; j < mma_warp_v_dim  ; j++ ) {
            RQ[0][0][0] = RD[0][j][0];
            RQ[1][0][0] = RD[0][j][1];
            RQ[0][0][1] = __shfl_sync((0xffffffff), RD[0][j][0],lane_id + 1 ,4);
            RQ[0][0][2] = __shfl_sync((0xffffffff), RD[0][j][0],lane_id + 2 ,4);
            RQ[0][0][3] = __shfl_sync((0xffffffff), RD[0][j][0],lane_id + 3 ,4);
            RQ[1][0][1] = __shfl_sync((0xffffffff), RD[0][j][1],lane_id + 1 ,4);
            RQ[1][0][2] = __shfl_sync((0xffffffff), RD[0][j][1],lane_id + 2 ,4);
            RQ[1][0][3] = __shfl_sync((0xffffffff), RD[0][j][1],lane_id + 3 ,4);
        
            if (lane_id % 4 == 0) {
                int reg_O_Br = warp_qp * mma_m * mma_warp_p + lane_id / 4;
                int store_lane_gmem_Q_Br = reg_O_Br + O_br_id * br;
                int warp_KV = 0;
                int reg_O_d  = warp_KV * mma_n * mma_warp_v_dim + j * mma_n;
                int store_lane_gmem_Q_d = reg_O_d;
                int gmem_0 = (O_gmem_offset +  store_lane_gmem_Q_Br * head_dim + 
                    store_lane_gmem_Q_d);
                int gmem_1 = (Q_gmem_offset + (store_lane_gmem_Q_Br + 8) * head_dim +
                    store_lane_gmem_Q_d);
                LDST128BITS(O[gmem_0]) = LDST128BITS(RQ[0][0][0]);
                LDST128BITS(O[gmem_1]) = LDST128BITS(RQ[1][0][0]);
            }
        }
    }
}


#include <cuda_fp16.h>
#include <cuda_runtime.h>
template <const int kHeadDim, const int kStage>
void launch_flash_attn_mma_stages_split_q_shared_kv_2(half * Q,
                                                    half * K,
                                                    half * V,
                                                    half * O,
                                                    const int QKV_batch,
                                                    const int QKV_head,
                                                    const int QKV_seqlen,
                                                    cudaStream_t stream
                                                    ) {
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;

  constexpr int kMmaTileSeqLenQ = 4;
  constexpr int kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = 4;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK =  8;
  constexpr int kWarpTileSeqLenP = 1;
  constexpr int kWarpTileHeadDimV =
      (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)); 
  constexpr int Br =
      kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; 
  constexpr int Bc =
      kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; 
  constexpr int kNumThreads =
      WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; 
  constexpr int kPadQ = 0;
  constexpr int kPadK = 0;
  constexpr int kPadV = 0;
 
  constexpr int kOStorageAccFloat32 = 0;


  constexpr int Q_tile_size = (Br * (kHeadDim + kPadQ));
  constexpr int K_tile_size = (Bc * (kHeadDim + kPadK));
  constexpr int V_tile_size = (Bc * (kHeadDim + kPadV));
  const int smem_max_size =
      (Q_tile_size + kStage * max(K_tile_size, V_tile_size)) * sizeof(half);

  assert(QKV_seqlen % max(Br, Bc) == 0); 


  dim3 grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head);
  dim3 block(kNumThreads); 

  printf("smem_max_size= %d\n", smem_max_size);
  printf("Launch config: grid(%d,%d,%d) block(%d,%d,%d)\n",
       grid.x, grid.y, grid.z, block.x, block.y, block.z);
  printf("seqlen=%d, head=%d, batch=%d\n", QKV_seqlen, QKV_head, QKV_batch);
  cudaFuncSetAttribute(
      flash_attn_mma_stages_split_q_shared_kv_kernel<
          kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
          kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      98304);

  flash_attn_mma_stages_split_q_shared_kv_kernel<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
      kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
      kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
      kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>
      <<<grid, block, 60000>>>(Q,
                                       K,
                                       V,
                                       O,
                                    QKV_batch,QKV_head,QKV_seqlen);

cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch error: %s\n", cudaGetErrorString(err));
}
}

template void launch_flash_attn_mma_stages_split_q_shared_kv_2<32, 2>(half*, half*, half*, half*, int, int, int, cudaStream_t);
template void launch_flash_attn_mma_stages_split_q_shared_kv_2<64, 2>(half*, half*, half*, half*, int, int, int, cudaStream_t);
template void launch_flash_attn_mma_stages_split_q_shared_kv_2<96, 2>(half*, half*, half*, half*, int, int, int, cudaStream_t);
template void launch_flash_attn_mma_stages_split_q_shared_kv_2<128,2>(half*, half*, half*, half*, int, int, int, cudaStream_t);
template void launch_flash_attn_mma_stages_split_q_shared_kv_2<160,2>(half*, half*, half*, half*, int, int, int, cudaStream_t);
template void launch_flash_attn_mma_stages_split_q_shared_kv_2<192,2>(half*, half*, half*, half*, int, int, int, cudaStream_t);
template void launch_flash_attn_mma_stages_split_q_shared_kv_2<256,2>(half*, half*, half*, half*, int, int, int, cudaStream_t);
