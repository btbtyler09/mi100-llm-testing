# mi100-llm-testing
This is a repository for documenting the setup and performance of MI100s in popular inference engines.

# vLLM
VLLM is supported on MI200 and MI300 series GPUS, but full support has not yet been added to MI100s.
Nevertheless, It is still possible to run VLLM on these GPUS. Currently lack of support for gfx908 in Flash Attention and AITER prevent building VLLM using the existing dockerfiles in the VLLM repo. I have included dockerfiles for building the base rocm container and VLLM. These build files exclude Flash Attention and AITER, allowing the build to complete with support for older GPUS. For these GPUS VLLM will use Triton Flash Attention, which is supported by the MI100, but may be unoptimized. I have also built these containers and pushed them to docker hub, so you can pull those containers directly.

## Build from source:
I'm providing a brief summary of this so you can build yourself. 

1. Pull the git repo for vLLM
2. Pull this repo
3. Copy the dockerfiles into vLLM
4. Build the base docker container for the MI100.
This container builds everything from source, so it takes time. The branches are specified in the dockerfile, so if you'd like to test new releases you can update those branches though you may run into compatibility issues.
```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg ARG_PYTORCH_ROCM_ARCH=gfx908 \
  -f Dockerfile.rocm_base \
  -t vllm-dev-mi100:base .
```
5. Build the vllm container
```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg ARG_PYTORCH_ROCM_ARCH=gfx908 \
  -f Dockerfile.rocm-mi100 \
  -t vllm-rocm-gfx908 .
```


## Pull the prebuilt container from docker hub
I may not be able to keep the container on docker hub up to date, but I will try to rebuild at each major release.

1. Pull the container from docker hub.
```bash
docker pull btbtyler09/vllm-rocm-gfx908
```
2. Start a new container with the image.

You will need to modify some of the instructions in this command. 
* I manually specify visible devices to make sure they are passed through correctly. render D128 represents the device in Node 0 if you run rocm-smi. The node numbers increment up from there. In this command I am specifying all 4 GPUs to be accessible in the container. You may also want to run multiple instances of the container with different GPUS, so 
you can specify a specific number for your use case.
* The huggingface cache folder is passed into the container to make your previously downloaded models accessible inside the container, and the HF_HOME environment variable is set to direct huggingface requests to that folder.

```bash
docker run -it \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd \
  --device=/dev/dri/renderD128 \
  --device=/dev/dri/renderD129 \
  --device=/dev/dri/renderD130 \
  --device=/dev/dri/renderD131 \
  --env VLLM_USE_V1=1 \
  --env HSA_OVERRIDE_GFX_VERSION=9.0.8 \
  --env VLLM_USE_TRITON_FLASH_ATTN=1 \
  --env HF_HOME=/huggingface \
  -v /home/{user}/.cache/huggingface:/huggingface \
  btbtyler09/vllm-rocm-gfx908 \
  bash
```
3. Run your model.
In this example I have specified a few extra instructions which may also need to be changed depending on you model of choice and hardware.
* The max-model-length may be increased if you have memory capacity to do so. If you are running on a single gpu, it may need to be decreased.
* tensor-parallel-size is set to 4 for running on 4 gpus. This needs to be a factor of 2, so you can remove it for running on a single GPU or change it to 2 or 8 depending on your available hardware.
* trust-remote-code is set for the Phi 4 model. Some models require this, but the Llama models do not. Phi 4 also has other dependencies you will need to install in the container before running this command. I believe they are scipy, peft, and backoff. If you run into an error, you should try the pip install it lists.
* kv-cache-dtype was a recent adition to vLLM. It can reduce memory usage for large context. I was testing with it enabled. **Doesn't work with V1 engine**
```bash
vllm serve microsoft/Phi-4-multimodal-instruct \
--gpu-memory-utilization 0.98 \
--guided-decoding-backend auto \
--max-model-len 32768 \
--tensor-parallel-size 4 \
--disable-log-requests \
--trust-remote-code 
```
## Supported Quantizations
It would be good to get some input on this. I have been able to quantize Llama-3.1-8B-Instruct to 4 and 8 bit using gptqmodel, but it took some trial and error. I haven't had any luck running GGUF models, and most models I have pulled from huggingface either refuse to run or spit out gibberish. I am trying to quantize Llama-3.3-70B, but I have run into issues with insufficient memory. I'm working on that, and will publish results with the 70b model as soon as I can get an 8-bit quantization up and running. The 124 GB isn't quite enough for running 70b models in FP16, but it should work well with 8 bit.

**Latest update (v0.8.5) seems to break gptq quant models**

My quantization of Llama-3.1 can be pulled from huggingface hub if you'd like to test it:
[btbtyler09/Llama-3.1-8B-Instruct-gptq-4bit](https://huggingface.co/btbtyler09/Llama-3.1-8B-Instruct-gptq-4bit)
