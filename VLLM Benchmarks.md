# VLLM Benchmarks for 4 x MI100 GPUS

## System Configuration
Lenovo P620 Workstation
Threadripper Pro 3745wx
256 GB (8 x 32GB) DDR4-2666MHz
4 x MI100 GPUs with Infinity Fabric Link

## VLLM Configuration
The following settings were used to launch the vllm server for each model. Thusfar minimal effort has gone into the configuration of vLLM for optimal performance. If options were changed or added for benchmarking they are listed as well. Some larger models have to be run with lower concurrency, and requests have been lowered to shorten benchmark time.

**10/27/2025 Update:**
* I am working on updating some benchmarks with ROCm 7+ and newer versions of vLLM. See most recent comparison with Qwen3 32B and new results for Qwen3 Next 80B.
* Qwen3 Next 80B has performance similar to Qwen3 32B (slower PP but faster TG). Likely my current reccomended model for 4 x MI100s.
* Vanilla vLLM builds for MI100s again. I have recent docker containers available at: https://hub.docker.com/r/btbtyler09/vllm-rocm-gfx908

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

## GPT-OSS 120B
* Now runs on latest docker container
```bash
vllm serve openai/gpt-oss-120b \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.94
```
**v0.11rc2.dev V1 Engine**
* 1 req/s
```bash
============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Request rate configured (RPS):           1.00      
Benchmark duration (s):                  102.94    
Total input tokens:                      102400    
Total generated tokens:                  5312      
Request throughput (req/s):              0.97      
Output token throughput (tok/s):         51.60     
Peak output token throughput (tok/s):    173.00    
Peak concurrent requests:                9.00      
Total Token throughput (tok/s):          1046.36   
---------------Time to First Token----------------
Mean TTFT (ms):                          198.17    
Median TTFT (ms):                        196.03    
P99 TTFT (ms):                           468.62    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          27.13     
Median TPOT (ms):                        22.83     
P99 TPOT (ms):                           63.44     
---------------Inter-token Latency----------------
Mean ITL (ms):                           26.24     
Median ITL (ms):                         20.51     
P99 ITL (ms):                            174.81    
==================================================
```

## Qwen 3 Next 80B Thinking
* jart25/Qwen3-Next-80B-A3B-Thinking-Int4-GPTQ
* vllm bench serve --base-url http://localhost:8000 --model jart25/Qwen3-Next-80B-A3B-Thinking-Int4-GPTQ --num-prompts 100 --request-rate x.x
* ROCm 7.0.2

**v0.11.1rc1.dev V1 Engine**
* 0.5 req/s
```bash
============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Request rate configured (RPS):           0.50      
Benchmark duration (s):                  203.55    
Total input tokens:                      102400    
Total generated tokens:                  12321     
Request throughput (req/s):              0.49      
Output token throughput (tok/s):         60.53     
Peak output token throughput (tok/s):    232.00    
Peak concurrent requests:                9.00      
Total Token throughput (tok/s):          563.59    
---------------Time to First Token----------------
Mean TTFT (ms):                          212.56    
Median TTFT (ms):                        207.05    
P99 TTFT (ms):                           359.75    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          29.36     
Median TPOT (ms):                        28.11     
P99 TPOT (ms):                           41.67     
---------------Inter-token Latency----------------
Mean ITL (ms):                           29.29     
Median ITL (ms):                         25.69     
P99 ITL (ms):                            184.61    
==================================================
```

* 1 req/s
```bash
============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Request rate configured (RPS):           1.00      
Benchmark duration (s):                  104.13    
Total input tokens:                      102400    
Total generated tokens:                  12409     
Request throughput (req/s):              0.96      
Output token throughput (tok/s):         119.17    
Peak output token throughput (tok/s):    348.00    
Peak concurrent requests:                15.00     
Total Token throughput (tok/s):          1102.57   
---------------Time to First Token----------------
Mean TTFT (ms):                          232.97    
Median TTFT (ms):                        213.22    
P99 TTFT (ms):                           401.61    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.10     
Median TPOT (ms):                        38.80     
P99 TPOT (ms):                           54.28     
---------------Inter-token Latency----------------
Mean ITL (ms):                           39.08     
Median ITL (ms):                         34.36     
P99 ITL (ms):                            190.46    
==================================================
```


## Qwen 3 32B
* Qwen/Qwen3-32B
* vllm bench serve --base-url http://localhost:8000 --model Qwen/Qwen3-32B --num-prompts 100 --request-rate 0.5
* ROCm 7.0.2

**v0.11.1rc2.dev323 V1 Engine**
```bash
============ Serving Benchmark Result ============
Successful requests:                     100       
Failed requests:                         0         
Request rate configured (RPS):           0.50      
Benchmark duration (s):                  206.73    
Total input tokens:                      102400    
Total generated tokens:                  12094     
Request throughput (req/s):              0.48      
Output token throughput (tok/s):         58.50     
Peak output token throughput (tok/s):    152.00    
Peak concurrent requests:                9.00      
Total Token throughput (tok/s):          553.84    
---------------Time to First Token----------------
Mean TTFT (ms):                          97.81     
Median TTFT (ms):                        97.55     
P99 TTFT (ms):                           129.19    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          53.45     
Median TPOT (ms):                        53.66     
P99 TPOT (ms):                           54.77     
---------------Inter-token Latency----------------
Mean ITL (ms):                           53.39     
Median ITL (ms):                         53.48     
P99 ITL (ms):                            62.88     
==================================================
```

### Previous results
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
