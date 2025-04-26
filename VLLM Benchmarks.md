# VLLM Benchmarks for 4 x MI100 GPUS

## System Configuration
Lenovo P620 Workstation
Threadripper Pro 3745wx
256 GB (8 x 32GB) DDR4-2666MHz
4 x MI100 GPUs with Infinity Fabric Link

## VLLM Configuration
The following settings were used to launch the vllm server for each model. Thusfar minimal effort has gone into the configuration of vLLM for optimal performance. If options were changed or added for benchmarking they are listed as well. Some larger models have to be run with lower concurrency, and requests have been lowered to shorten benchmark time.

```bash
vllm serve <model> \
        --swap-space 16 \
        --gpu-memory-utilization 0.98 \
        --guided-decoding-backend outlines \
        --max-model-len 32768 \
        --tensor-parallel-size 4 \
        --disable-log-requests \
        --trust-remote-code \
        --kv-cache-dtype fp8
```
```bash
python benchmarks/benchmark_serving.py --dataset-name=random --model <model> --max-concurrency 50
```
## IBM Granite 3.3 8b Instruct
* ibm-granite/granite-3.3-8b-instruct
* max-concurrency 25
* kv-cahce-dtype fp16

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
```
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
```
============ Serving Benchmark Result ============
Successful requests:                     1000      
Benchmark duration (s):                  201.59    
Total input tokens:                      1024000   
Total generated tokens:                  120912    
Request throughput (req/s):              4.96      
Output token throughput (tok/s):         599.78    
Total Token throughput (tok/s):          5679.34   
---------------Time to First Token----------------
Mean TTFT (ms):                          91823.28  
Median TTFT (ms):                        83208.08  
P99 TTFT (ms):                           175712.39 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          406.47    
Median TPOT (ms):                        347.66    
P99 TPOT (ms):                           822.77    
---------------Inter-token Latency----------------
Mean ITL (ms):                           336.52    
Median ITL (ms):                         271.13    
P99 ITL (ms):                            2121.01   
==================================================
```
