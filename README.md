This repository implements FlashAttention and optimizes it based on the project from LeetCUDA. The main optimizations include:

Making global memory accesses contiguous, achieving 100% sector utilization(fla2_combine.cu):

  sector utilization = L2 theoretical sectors global / L2 theoretical sectors global ideal = 100%
Applying swizzling across blocks(fla2_swizzleblock.cu).

Both of the above optimizations are implemented for the Q matrix, serving as a comparison against the K/V matrices.
