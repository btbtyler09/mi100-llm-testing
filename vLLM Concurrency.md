# Concurrency Benchmarking

## System Configuration
* Lenovo P620 Workstation
* Threadripper Pro 3745wx
* 256 GB (8 x 32GB) DDR4-2666MHz
* 4 x MI100 GPUs with Infinity Fabric Link (XGMI)
* GPUs Power Limit Set to 290W (default limit)

## VLLM Configuration
* vLLM Version: 0.9.1.dev59+gb6a6e7a52 (commit b6a6e7a)
The following settings were used to launch the vllm server for each model.

```bash
vllm serve <model> \
        --gpu-memory-utilization 0.98 \
        --max-model-len 8192 \
        --tensor-parallel-size 4 \
        --disable-log-requests \
```
```bash
python benchmarks/benchmark_serving.py --dataset-name=random --model <model> --max-concurrency <concurrency> --num-prompts 100
```

## Qwen 3 32B 4-bit GPTQ
This model from kaitchup seems to work well with the recent versions of vLLM
https://huggingface.co/kaitchup/Qwen3-32B-autoround-4bit-gptq

### Output Token Throughput (tok/s)
|GPU Quantity|1 Concurrency|2 Concurrency|6 Concurrency|8 Concurrency|10 Concurrency|50 Concurrency|100 Concurrency|
|---|---|---|---|---|---|---|---|
|4|28.7|53|114.5|162.3|172|377|488|

### Total Token Throughput (tok/s)
|GPU Quantity|1 Concurrency|2 Concurrency|6 Concurrency|8 Concurrency|10 Concurrency|50 Concurrency|100 Concurrency|
|---|---|---|---|---|---|---|---|
|4|273.5|505.3|1088.7|1542.5|1638|3583|4654|

### Mean Time to First Token (ms)
|GPU Quantity|1 Concurrency|2 Concurrency|6 Concurrency|8 Concurrency|10 Concurrency|50 Concurrency|100 Concurrency|
|---|---|---|---|---|---|---|---|
|4|63|110|169|190.1|213|750.5|1145|

### Notes:
* My machine is thermal limited. Running these benchmarks at the full power limit of 290W can lead to thermal throttling. For these tests I believe I was able to avoid that by keeping runtimes low.


