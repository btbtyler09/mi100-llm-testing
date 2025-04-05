# VLLM Benchmarks for 4 x MI100 GPUS

## System Configuration
Lenovo P620 Workstation
Threadripper Pro 3745wx
256 GB (8 x 32GB) DDR4-2666MHz
4 x MI100 GPUs with Infinity Fabric Link

## VLLM Configuration
The following settings were used to launch the vllm server for each model. Thusfar minimal effort has gone into the configuration of vLLM for optimal performance.

```bash
vllm serve <model> \
        --swap-space 16 \
        --gpu-memory-utilization 0.97 \
        --guided-decoding-backend outlines \
        --max-model-len 32768 \
        --tensor-parallel-size 4 \
        --disable-log-requests \
        --trust-remote-code \
        --kv-cache-dtype fp8
```
```bash
python benchmarks/benchmark_serving.py --dataset-name=random --model <model>
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
Benchmark duration (s):                  277.88    
Total input tokens:                      1024000   
Total generated tokens:                  120784    
Request throughput (req/s):              3.60      
Output token throughput (tok/s):         434.67    
Total Token throughput (tok/s):          4119.74   
---------------Time to First Token----------------
Mean TTFT (ms):                          125772.36 
Median TTFT (ms):                        115250.72 
P99 TTFT (ms):                           244852.99 
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          536.72    
Median TPOT (ms):                        488.96    
P99 TPOT (ms):                           1213.38   
---------------Inter-token Latency----------------
Mean ITL (ms):                           463.36    
Median ITL (ms):                         355.74    
P99 ITL (ms):                            3266.19   
==================================================
```