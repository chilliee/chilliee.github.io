---
layout: post
title:  "Dense Matrix Transposition on Modern Computer Architectures"
date:   2021-05-15 21:09:48 -0400
---
Dense matrix transposition is a widespread problem in numerical linear algebra. It plays a significant role in most advanced technologies today like Deep Learning, Computer Vision, Autonomous Driving, etc. Anyone who learns a little C and linear algebra can solve this problem on a silver platter:
```c
// transpose an MxN matrix A into an NxM matrix B
for (int i = 0; i < N; ++i) {
  for (int j = 0; j < M; ++j) {
    B[i][j] = A[j][i]
  }
}
```
Beginners in the computer architecture area may comment: "Uh-huh, this operation just moves data from one place to another, nothing interesting! Optimizing this should be the work of people who design fast main memory!"

That's true indeed, considering the gap between processor and memory speeds in computers today. However, when you try to compare this implementation with the copy operation, you will find it is much slower than what you might expect (for floating point matrix of 1024x1024):

<table>
  <tr>
    <td></td>
    <td align="center" colspan="2">Effective Bandwidth (GB/s)</td>
  </tr>
  <tr>
    <td align="center">Routine</td>
    <td align="center">i5-9600K</td>
    <td align="center">RTX-2070</td>
  </tr>
  <tr>
    <td align="center">Copy</td>
    <td align="center">36.7</td>
    <td align="center">255.45</td>
  </tr>
  <tr>
    <td align="center">Transpose</td>
    <td align="center">2.0</td>
    <td align="center">149.9</td>
  </tr>
</table>

In fact, we do have techniques to write faster code for this problem with the power of instruction level parallelism, multi/many-core architecture, and cache. We will explore some of them in this blog for modern computer architectures like CPUs and GPUs.

### Instruction Level Parallelsim
Modern processors tend to use an instruction set that is quite similar to MIPS in which there are 5 stages in the pipeline:

![MIPS pipeline](https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Fivestagespipeline.png/400px-Fivestagespipeline.png)
- **IF**: Instruction fetch
- **ID**: Instruction decode
- **EX**: Execute
- **MEM**: Memory access
- **WB**: Writeback

Just as shown in the image above, we can pipeline different instructions to parallelize code in a single core. Specifically for our problem, if previous instruction is at the MEM stage - since memeory access is typically much slower than previous stages, we can ask next instruction to fetch the instruction, perform the decoding or even execute. This could be manually optimized by:
```c
// unrolling the inner loop to utilize ILP
for (int i = 0; i < N; ++i)
  for (int j = 0; j < N; j += 4) {
      B[i][j]   = A[j][i];
      B[i][j+1] = A[j+1][i];
      B[i][j+2] = A[j+2][i];
      B[i][j+3] = A[j+3][i];
  }
```
For some compilers, there exist directives to lower this burden. For example, we use `#pragma GCC unroll n` with GCC or `#pragma loop( hint_parallel(n) )` with MSVC. This also happens in GPU programming like CUDA, in which case we use `#pragma unroll`

### Multi/Many-core Architecture Acceleration
Though increasing the single-core capability has long been one of the major goals for modern processors, the breakdown of Dennard scaling forces the processor design to be more focused on chips with multiple cores. And for GPUs, the more cores you have, the more powerful your device is. This design makes it possible to introduce explicit parallel programming into CPUs and GPUs.

For CPUs, we can use OpenMP to control this level of parallelism by adding OpenMP directives to the code:
```c
// with OpenMP
for (int i = 0; i < N; ++i) {
  #pragma omp parallel for
  for (int j = 0; j < M; ++j) {
    B[i][j] = A[j][i]
  }
}
```
In fact, OpenMP is a fork-join model which creates threads running concurrently to solve the problem. I use the `for` clause here just for simplicity.

And for GPUs, we use CUDA to expose the parallelism in manycore platforms:
```cuda
__global__ void transpose(const float *A, float *B, const int M, const int N);

void main()
{
...
  transpose<<<grid, thread>>>(A, B, M, N);
...
}
```
in which tasks are separated into grids and then separated into threads.

### Cache or Using Memory Hierarchy
We've discussed two aspects of modern computer architectures to give potential optimization on our initial implementation. However, it still cannot interpret why direct copy runs much faster than transposition. The main reason is that we neglect the memory hierarchy - the access time of each data element is not the same! To be more specific, the first time we access the `A` matrix, we will load more than one item into a small-size memory that has a much faster access speed than DRAM main memory. Then next time we access the following element, we can directly fetch data from this faster memory instead of going to the DRAM. This small memory is called cache.

<div style="text-align: center">
<img src="/assets/images/cache_cpu.png" alt="cpu_cache" style="width:400px;"/>
</div>

However, since the cache size is much smaller than the main memory, the expected data will not always be in the cache. So when there is a miss in the cache, we still need to access the main memory and get no benefit from the cache. To minimize cache misses, caches on modern processors are usually designed for locality:
- **Temporal locality**: If at one point a particular memory location is referenced, then it is likely that the same location will be referenced again in the near future. There is temporal proximity between adjacent references to the same memory location. In this case, it is common to make efforts to store a copy of the referenced data in faster memory storage to reduce the latency of subsequent references. Temporal locality is a special case of spatial locality (see below), namely when the prospective location is identical to the present location.
- **Spatial locality**: If a particular storage location is referenced at a particular time, then it is likely that nearby memory locations will be referenced in the near future. In this case, it is common to attempt to guess the size and shape of the area around the current reference, for which it is worthwhile to prepare faster access for subsequent reference.

For the copy operation, we only access contiguous memory, and cache misses only happen when the first time we access the data or when the cache is full. On the other hand, for transposition, even we contiguously access `B` matrix, the `A` matrix is still accessed column by colunm (2D arrays or matrices in C are in row-major). When you access `A[i][0]`, for example, you will load `A[i][0]` until `A[i][cacheline]` to the cache. And thus, if you access, say K rows, there are Kxcacheline data in the cache, and if the matrix is large enough - like 1024x1024 in the above example - the cache can be easily filled up and needs to be replaced. After you access the first column, what remains in the cache will be `A[1024-K:1023][0:cacheline-1]` and there is no benefit when you try to access the second column - no hit in the cache!

One solution to mitigate cache misses in our problem is to use tiling:
<div style="text-align: center">
<img src="/assets/images/tiling.png" alt="tiling" style="width:500px;"/>
</div>
By tiling, we could push blocks of data into the cache and reduce cache misses. This could be easily implemented as:
```c
// tiling
for (int i = 0; i < N; i += blk_l2) {
  for (int j = 0; j < M; j += blk_l2) {
    for (int ii = i; ii < i + blk_l2; ++ii) {
      for (int jj = j; jj < j + blk_l2; ++jj) {
        B[ii][jj] = A[jj][ii]
      }
    }
  }
}
```
And since there are multiple levels of caches, we could also add more levels of tiling to optimize the code:
```c
// tiling - 2 level of cache
for (int i = 0; i < N; i += blk_l2) {
  for (int j = 0; j < M; j += blk_l2) {
    for (int ii = i; ii < i + blk_l2; ii += blk_l1) {
      for (int jj = j; jj < j + blk_l2; jj += blk_l1) {
        for (int iii = ii; iii < ii + blk_l1; ++iii) {
          for (int jjj = jj; jjj < jj + blk_l1; ++jjj) {
            B[iii][jjj] = A[jjj][iii]
          }
      }
    }
  }
}
```

The same thing also happens on GPUs, where we could explicitly configure the shared memory as the cache:
```cuda
__global__ void transpose(const float *A, float *B, const int M, const int N){
  __shared__ sm[TILE_SIZE][TILE_SIZE];

  // load data into the shared memory
  ...
  __syncthreads()

  // transpose
  ...
}
```
One difference on GPU programming is that the shared memory is concurrently accessed by many threads within a block and we can coalesce the accesses though shared memory. Moreover, the shared memory that can be accessed in parallel is divided into banks, and if 2 threads tend to access the same bank, the 2 accesses will be done sequentially. So a further optimization on GPUs could configure the shared memory size to be `sm[TILE_SIZE][TILE_SIZE+1]` if the `TILE_SIZE` is multiple of `warpSize`.

### What We've Done So Far
Now let's back to the performance of matrix transposition in the beginning:
<table>
  <tr>
    <td></td>
    <td align="center" colspan="2">Effective Bandwidth (GB/s)</td>
  </tr>
  <tr>
    <td align="center">Routine</td>
    <td align="center">i5-9600K</td>
    <td align="center">RTX-2070</td>
  </tr>
  <tr>
    <td align="center">Copy</td>
    <td align="center">36.7</td>
    <td align="center">255.45</td>
  </tr>
  <tr>
    <td align="center">Transpose</td>
    <td align="center">2.0</td>
    <td align="center">149.9</td>
  </tr>
  <tr>
    <td align="center">Loop Unrolling + Core Prallelism + Cache</td>
    <td align="center">24.7</td>
    <td align="center">271.3</td>
  </tr>
</table>
We got more than x12 speedup on CPU and near x2 speedup on GPU. Note that on the GPU, we get even larger bandwidth than the copy. This is because we do not use shared memory in copy and can be optimized in practice.

Modern computer architectures are usually quite complicated systems, and remember to always take aspects of the underlying designs into considerations when writing code.