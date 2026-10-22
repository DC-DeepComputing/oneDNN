GPU Developer Mode Controls {#dev_guide_gpu_dev_mode}
=====================================================

oneDNN provides additional controls for Intel GPU kernel development and
performance exploration. These controls are intended for oneDNN developers.
They are not part of the stable API and may change without notice.

## Build-time Controls

The controls are available only if the library is built with
`ONEDNN_DEV_MODE=ON`. In other builds they do not take any effect.

| CMake Option    | Supported values (defaults in bold) | Description                     |
|:----------------|:------------------------------------|:--------------------------------|
| ONEDNN_DEV_MODE | **OFF**, ON                         | Enables developer-only controls |

## GPU Logging

The `xe=<level>` token of the `ONEDNN_VERBOSE` environment variable enables
Intel GPU specific logs. These logs are independent of the generic
[verbose mode](@ref dev_guide_verbose) output and can be combined with other
verbose flags, e.g. `ONEDNN_VERBOSE=dispatch,xe=debug`.

| Level      | Description                                        |
|:-----------|:---------------------------------------------------|
| **off**    | GPU logs are disabled (default)                    |
| warning    | Warnings                                           |
| suggestion | Suggestions for better performance                 |
| info       | Implementation decisions, e.g. the selected kernel |
| debug      | Details of the implementation decisions            |
| perf       | Performance-related details                        |
| trace      | All messages                                       |
| all        | Same as `trace`                                    |

Each level includes the messages of the levels above it. An integer value on
the `debuginfo` scale is also accepted.

### Example: GEMM Kernel Selection

With `xe=info`, the JIT GEMM implementation prints the kernel it uses. The
examples on this page are from an Intel Arc B580 GPU:

~~~sh
    $ ONEDNN_VERBOSE=xe=info ./benchdnn --matmul --engine=gpu --dt=f16 1024x1024:1024x1024
[ INFO][...] [ 658] use 0/31 gemm HH[SH] N@8N@8N 64 32 1 0 av32+m32@48 am32+S32@64 aB wg 4x8 xaf st vav hi pt sb32 bk0 sn grf256 sys sr br afb rr
[ INFO][...] [ 658] final gemm HH[SH] N@16N@16N 64 32 1 0 aV32#16,0{cc}+M32,32@48{cc} aM32#16,0{cc}+S32,32@64{cc} aB#16,0{cc/bb} wg 4x8 sys kc1 xaf st k32 grf256 raxehpc sn vav di sr bf cc rr sb32 np bm16777216 bn16777216 bk16777216
~~~

The bracketed number is the catalog entry index. `use i/n` shows the rank of
the selected entry among the `n` sorted candidates. The `use` line contains
the kernel string, which can be passed back to the library as a
[kernel override](#gemm-kernel-override). The `final` line shows the
strategy after all internal adjustments.

With `xe=debug`, the implementation also prints each candidate it considers
with its model score and estimated time, and the reason for each candidate it
skips.

## GEMM Kernel Override

The JIT GEMM implementation can use a specified kernel instead of the
selected catalog entry. The override accepts one of these values:

| Value         | Description                                      |
|:--------------|:-------------------------------------------------|
| `N`           | Uses the candidate with rank `N` (`use N/n ...`) |
| kernel string | Uses the kernel described by the string          |

The override can be set in these ways:

| Method                                             | Scope                |
|:---------------------------------------------------|:---------------------|
| benchdnn `--gpu-kernel=STR` option (matmul driver) | One benchdnn problem |
| `dnnl_impl_gpu_intel_set_kernel_override(attr, s)` | One primitive        |
| `GEMM_KERNEL` environment variable                 | All GEMM problems    |

The benchdnn option also accepts a rank range `A-B` and `all`. Refer to the
[matmul driver](https://github.com/uxlfoundation/oneDNN/blob/main/tests/benchdnn/doc/driver_matmul.md)
documentation for details. The internal
`dnnl_impl_gpu_intel_get_kernel_count(pd, &count)` function returns the
number of candidates for a primitive descriptor. The environment variable
takes precedence over the attribute.

### Example: Rank Override

~~~sh
    $ ONEDNN_VERBOSE=xe=info ./benchdnn --matmul --engine=gpu --dt=f16 --gpu-kernel=1 1024x1024:1024x1024
[ INFO][...] [ 657] use 1/31 pin gemm HH[SH] N@8N@8N 64 40 1 0 av16+m32@72 am32+m32@64 aB wg 4x8 xaf vav hi pt sr br sb64 bk0 sn grf256 sys afb
...
~~~

### Example: Kernel String Override

Copy the part of the `use` line that starts at `gemm`. Edit the strategy
tokens as necessary, e.g. change the workgroup shape from `wg 4x8` to `wg 8x4`:

~~~sh
    $ ONEDNN_VERBOSE=xe=info ./benchdnn --matmul --engine=gpu --dt=f16 \
      --gpu-kernel="gemm HH[SH] N@8N@8N 64 32 1 0 av32+m32@48 am32+S32@64 aB wg 8x4 xaf st vav hi pt sb32 bk0 sn grf256 sys sr br afb rr" \
      1024x1024:1024x1024
[ INFO][...] [override] use gemm HH[SH] N@8N@8N 64 32 1 0 av32+m32@48 am32+S32@64 aB wg 8x4 xaf st vav hi pt sb32 bk0 sn grf256 sys sr br afb rr
...
~~~

The kernel string has the following format:

~~~
    gemm <precisions> <layouts> <unroll_m> <unroll_n> <alpha> <beta> <strategy>
~~~

Tokens before `gemm` are ignored. The override applies only to a compatible
problem:

- The external data types and the layouts of A, B, and C must match the
  problem. f32 and tf32 are interchangeable for A and B.
- The alignments in the string must not exceed the actual buffer alignments.
- A fixed alpha or beta must match the problem. Use `-` to accept any value.

If the override does not apply or cannot be built, primitive creation fails.
Use `ONEDNN_VERBOSE=dispatch` to see the reason, e.g.
`kernel override rejected: invalid external A data type`.

A problem that the library splits into a GEMM and separate post-ops passes a
kernel string to the inner GEMM, which has an f32 C. A rank is rejected for
such a problem. The systolic GEMM implementation rejects all overrides.

@warning
The override can change the internal data types of the computation. An
incompatible precision can produce inaccurate results.
