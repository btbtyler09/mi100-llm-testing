# GLM-5.3-Flash on eight MI100 GPUs: measured performance history

These are historical measurements of the complete 45-layer W4A16 text model at TP8. The model retained packed INT4/group-128 experts, BF16 activations/main KV and FP32 recurrent state. Different stages used different prompts, capacities and timing methods; their rates are not one controlled end-to-end speedup series.

For the portable current configuration and new measurements, see the [native K=1 report](../../benchmark_GLM-5.3-Flash-Uncensored-W4A16_mtp.md).

## September 29: approximately 2.8 to 30 tokens/s

Controlled sequential requests used exact token-ID inputs, fixed output lengths, temperature zero, three warmups and an 8,192-token context limit.

| Input/output tokens | Original median decode | Optimized median decode | Repeated optimized decode |
|---|---:|---:|---:|
| 128/128 | 2.837 tokens/s | 30.358 tokens/s | 30.353 tokens/s |
| 4,096/128 | 2.725 tokens/s | 30.223 tokens/s | 30.209 tokens/s |

The short case improved 10.7× in decode, with about 90.7% lower estimated TPOT. Short-prompt median TTFT increased from 1.455 seconds to 1.674–1.708 seconds. The 4K case's median TTFT fell from 5.279 seconds to 3.676–4.066 seconds. The measurements establish a decode gain with a short-prompt first-token tradeoff.

The implementation combined:

- Fused gfx908 mHC kernels preserving the 20 Sinkhorn iterations and accepted numerical boundaries.
- Small-token projection/RMS and stream-reduction/norm fusion.
- BF16-input/FP32-accumulation router GEMV with original top-8 routing semantics.
- Decode graph replay and one CPU thread per worker.
- Removal of diagnostic finite-logit synchronization from timed serving.

The first mHC fusion alone measured 5.539 tokens/s in an intermediate eager run. The final result included multiple changes and cannot be attributed entirely to graphs or any isolated kernel.

The short trace fell from 15,305 GPU kernels per decode step per rank to 2,280, an 85.1% reduction. Profiling occurred after timing; profiled spans were not treated as benchmark throughput. The historical release passed 198 component cases with two skips, actual-weight/engine checks, API checks and a ten-minute soak.

## October 7: approximately 30 to 34 tokens/s

The later candidate enabled the single-group expert selector, direct packed-W4 expert execution and a linear tool-result renderer.

| C1 screening shape | Cold decode | Warm decode | Cold TTFT | Warm TTFT |
|---|---:|---:|---:|---:|
| 128/128 | 33.623 tokens/s | 33.492 tokens/s | 1.799 s | 1.726 s |
| 4,096/128 | 34.452 tokens/s | 34.428 tokens/s | 6.583 s | 1.317 s |

Post-start 4K/128 checks measured 33.925/34.012 tokens/s; the warm request reused 3,840 prompt tokens. These were candidate screening/deployment checks. Fresh-boot ABBA qualification and the final soak were not completed for that historical candidate.

## October 7: native MTP K=1

Registration/configuration changes activated the checkpoint's existing prediction head. It used the stored weights, with standard target verification and one speculative token; no separate draft checkpoint or relaxed acceptance was introduced.

The coding fixture had 34 input tokens and 128 returned tokens:

| Run | Decode tokens/s |
|---|---:|
| MTP-off control | 33.4687 |
| MTP-off repeat | 33.3980 |
| Initial cold MTP, 2K capacity and one admitted sequence | 36.1668 |
| Serving-profile K=1, first repeat | 49.4553 |
| Serving-profile K=1, second repeat | 49.3089 |
| Deployed K=1 | 48.9703 |

The serving-profile median was 49.3089 tokens/s versus 33.4334 for the controls, a 47.48% increase. The deployed request had 0.3588-second TTFT and 2.9524-second end-to-end latency. The original control had 0.3117-second TTFT and 4.1065-second end-to-end latency.

Later MTP samples followed warmup probes and used auto-fit capacity, two-sequence admission and graphs `[1,2,4]`. Those changes were not individually isolated. The rise from the cold 36.17 result to approximately 49 cannot be assigned to graph sizes alone.

Two simultaneous 128-token streams measured 36.5423 and 36.5462 tokens/s per call, with 3.4750 seconds of output overlap. This was one focused pair, not a concurrency sweep or an independently measured aggregate rate.

Model-loading allocation increased to approximately 23.51 GiB per GPU. The auto-fit prompt-plus-output capacity fell from 661,760 to 387,200 tokens. Prefix reuse remained functional, but the final 640-token block was reserved for draft lookahead: a 1,251-token prefix fixture had no hit, while a 1,571-token fixture reused 640 tokens.

The initial MTP sample proposed 65 draft tokens and accepted 62, including three rejections. A greedy token disagreement was also observed in MTP-off repeats on close alternatives. This did not establish universal parity, and the short tests did not qualify deeper speculation or long-context state handling.
