# VLLM Benchmarks for 4 x MI100 GPUS

## System Configuration
Lenovo P620 Workstation
Threadripper Pro 3745wx
256 GB (8 x 32GB) DDR4-2666MHz
4 x MI100 GPUs with Infinity Fabric Link

## VLLM Configuration
The following settings were used to launch the vllm server for each model. Thusfar minimal effort has gone into the configuration of vLLM for optimal performance. If options were changed or added for benchmarking they are listed as well. Some larger models have to be run with lower concurrency, and requests have been lowered to shorten benchmark time.

**Very Important to set VLLM_USE_V1=1 in v0.8.5. Otherwise performance is terrible**
**kv-cache-dtype of FP8 is not supported in V1 engine**
**gptq quantization doesn't seem to work on V1 engine with v0.8.5, so I can't test any of those quants**
**Native FP8 Models also don't work.**

```bash
vllm serve <model> \
        --gpu-memory-utilization 0.98 \
        --max-model-len 32768 \
        --tensor-parallel-size 4 \
        --disable-log-requests
```
```bash
python benchmarks/benchmark_serving.py --dataset-name=random --model <model> --max-concurrency 50
```
## Qwen 3 32B
* Qwen/Qwen3-32B
* max-concurrency 5
* num-prompts 100

**v0.8.5 V1 Engine**
```bash
============ Serving Benchmark Result ============
Successful requests:                     100       
Benchmark duration (s):                  205.09    
Total input tokens:                      102400    
Total generated tokens:                  11961     
Request throughput (req/s):              0.49      
Output token throughput (tok/s):         58.32     
Total Token throughput (tok/s):          557.61    
---------------Time to First Token----------------
Mean TTFT (ms):                          632.53    
Median TTFT (ms):                        475.08    
P99 TTFT (ms):                           1491.74   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          83.03     
Median TPOT (ms):                        80.32     
P99 TPOT (ms):                           103.10    
---------------Inter-token Latency----------------
Mean ITL (ms):                           79.92     
Median ITL (ms):                         70.30     
P99 ITL (ms):                            388.75    
==================================================
```

## Qwen 3 30B A3B
* Qwen/Qwen3-30B-A3B
* max-concurrency 16
* num-prompts 250

**v0.8.5 V1 Engine**
```bash
============ Serving Benchmark Result ============
Successful requests:                     250       
Benchmark duration (s):                  98.74     
Total input tokens:                      256000    
Total generated tokens:                  32000     
Request throughput (req/s):              2.53      
Output token throughput (tok/s):         324.07    
Total Token throughput (tok/s):          2916.66   
---------------Time to First Token----------------
Mean TTFT (ms):                          340.43    
Median TTFT (ms):                        263.68    
P99 TTFT (ms):                           1261.22   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          46.07     
Median TPOT (ms):                        46.23     
P99 TPOT (ms):                           48.85     
---------------Inter-token Latency----------------
Mean ITL (ms):                           46.07     
Median ITL (ms):                         38.05     
P99 ITL (ms):                            173.50    
==================================================
```

## IBM Granite 3.3 8b Instruct
* ibm-granite/granite-3.3-8b-instruct
* max-concurrency 25

**v0.8.5 V1 Engine**
```bash
============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  131.10    
Total input tokens:                      512000    
Total generated tokens:                  60018     
Request throughput (req/s):              3.81      
Output token throughput (tok/s):         457.81    
Total Token throughput (tok/s):          4363.30   
---------------Time to First Token----------------
Mean TTFT (ms):                          278.67    
Median TTFT (ms):                        193.69    
P99 TTFT (ms):                           1516.23   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          53.50     
Median TPOT (ms):                        53.93     
P99 TPOT (ms):                           150.13    
---------------Inter-token Latency----------------
Mean ITL (ms):                           52.09     
Median ITL (ms):                         32.88     
P99 ITL (ms):                            209.92    
==================================================
```

**0.8.3**
```bash
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  498.37    
Total input tokens:                      1024000   
Total generated tokens:                  119497    
Request throughput (req/s):              2.01      
Output token throughput (tok/s):         239.78    
Total Token throughput (tok/s):          2294.49   
---------------Time to First Token----------------
Mean TTFT (ms):                          1040.36   
Median TTFT (ms):                        247.92    
P99 TTFT (ms):                           7139.91   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          99.63     
Median TPOT (ms):                        75.27     
P99 TPOT (ms):                           285.20    
---------------Inter-token Latency----------------
Mean ITL (ms):                           95.96     
Median ITL (ms):                         28.52     
P99 ITL (ms):                            2635.79   
==================================================
```

## Cogit0 v1 Preview Llama 3.3 70b GPTQ 8 bit
* btbtyler09/cogito-v1-preview-llama-70B-gptq-8bit
* --max-concurrency 6
* --num-prompts 50

**v0.8.3 V0 Engine**
```bash
============ Serving Benchmark Result ============
Successful requests:                     50        
Benchmark duration (s):                  212.43    
Total input tokens:                      51200     
Total generated tokens:                  5945      
Request throughput (req/s):              0.24      
Output token throughput (tok/s):         27.99     
Total Token throughput (tok/s):          269.00    
---------------Time to First Token----------------
Mean TTFT (ms):                          6078.67   
Median TTFT (ms):                        6250.83   
P99 TTFT (ms):                           8916.73   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          182.00    
Median TPOT (ms):                        165.98    
P99 TPOT (ms):                           814.08    
---------------Inter-token Latency----------------
Mean ITL (ms):                           160.99    
Median ITL (ms):                         98.83     
P99 ITL (ms):                            1994.50   
==================================================
```

## Cogito v1 Preview Qwen 32B GPTQ 8 bit
* btbtyler09/cogito-v1-preview-qwen-32B-gptq-8bit
* --max-concurrency 18

**v0.8.3 V0 Engine**
```bash
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  1488.41   
Total input tokens:                      1024000   
Total generated tokens:                  126605    
Request throughput (req/s):              0.67      
Output token throughput (tok/s):         85.06     
Total Token throughput (tok/s):          773.04    
---------------Time to First Token----------------
Mean TTFT (ms):                          11258.26  
Median TTFT (ms):                        11539.83  
P99 TTFT (ms):                           13924.52  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          125.05    
Median TPOT (ms):                        119.65    
P99 TPOT (ms):                           207.67    
---------------Inter-token Latency----------------
Mean ITL (ms):                           123.53    
Median ITL (ms):                         112.49    
P99 ITL (ms):                            125.29    
==================================================
```

## Gemma 3 27b
* Model won't run in vLLM 0.8.5, need to investigate.
* This model ran in older versions of vLLM, but had issues with outputting gibberish at longer context.

**v0.8.3 V0 Engine**
```bash
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  547.00    
Total input tokens:                      1024000   
Total generated tokens:                  45250     
Request throughput (req/s):              1.83      
Output token throughput (tok/s):         82.72     
Total Token throughput (tok/s):          1954.77   
---------------Time to First Token----------------
Mean TTFT (ms):                          238989.80 
Median TTFT (ms):                        221321.34 
P99 TTFT (ms):                           485237.32 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          4762.90   
Median TPOT (ms):                        2354.20   
P99 TPOT (ms):                           29628.82  
---------------Inter-token Latency----------------
Mean ITL (ms):                           2073.37   
Median ITL (ms):                         763.79    
P99 ITL (ms):                            20693.33  
==================================================
```

## Phi 4 Multimodal vllm results
* updated results for vLLM 0.8.5
* --max-concurrency=20
* --trust-remote-code

**v0.8.5 V1 Engine**
```bash
============ Serving Benchmark Result ============
Successful requests:                     500       
Benchmark duration (s):                  84.45     
Total input tokens:                      512000    
Total generated tokens:                  60962     
Request throughput (req/s):              5.92      
Output token throughput (tok/s):         721.90    
Total Token throughput (tok/s):          6784.87   
---------------Time to First Token----------------
Mean TTFT (ms):                          243.32    
Median TTFT (ms):                        198.94    
P99 TTFT (ms):                           1143.85   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          31.54     
Median TPOT (ms):                        31.29     
P99 TPOT (ms):                           33.89     
---------------Inter-token Latency----------------
Mean ITL (ms):                           31.22     
Median ITL (ms):                         22.33     
P99 ITL (ms):                            114.07    
==================================================
```
