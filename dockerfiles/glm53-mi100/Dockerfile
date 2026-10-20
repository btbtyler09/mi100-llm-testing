FROM btbtyler09/vllm-rocm-gfx908@sha256:3e9d1e5a128413c6f038d6ae7a073b3ca99fda3483303cdfe1791385ef8ae013
COPY source-lock.json overlay-manifest.json base-files.sha256 LICENSE /opt/glm53-build/
COPY scripts/verify_base.py /opt/glm53-build/verify_base.py
RUN python3 /opt/glm53-build/verify_base.py
COPY overlay/vllm/ /usr/local/lib/python3.12/dist-packages/vllm/
COPY scripts/ /opt/glm53-build/scripts/
COPY tests/ /opt/glm53-build/tests/
RUN python3 /opt/glm53-build/scripts/verify_image.py --cpu-only
ENV VLLM_GFX908_MOE_HIP=0 VLLM_GFX908_MOE_FP16_COMPUTE=0 VLLM_GFX908_SHARED_AS_EXPERT=0
ENV VLLM_USE_V2_MODEL_RUNNER=1
ENV VLLM_INDEXER_PAGED_Q_BF16=0 VLLM_INDEXER_LOGITS_MAXNREG=0 VLLM_INDEXER_LOGITS_FACTOR_K_SCALE=0
ENTRYPOINT ["vllm", "serve"]
