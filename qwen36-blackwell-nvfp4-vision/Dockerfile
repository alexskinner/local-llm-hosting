FROM vllm/vllm-openai:cu130-nightly

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8000

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]

CMD ["AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP", \
     "--served-model-name", "96gbBigModel", \
     "--limit-mm-per-prompt", "{\"image\":4,\"video\":2}", \
     "--quantization", "modelopt_fp4", \
     "--tensor-parallel-size", "1", \
     "--max-model-len", "262144", \
     "--max-num-seqs", "64", \
     "--max-num-batched-tokens", "16384", \
     "--block-size", "32", \
     "--gpu-memory-utilization", "0.93", \
     "--enable-prefix-caching", \
     "--enable-chunked-prefill", \
     "--enable-auto-tool-choice", \
     "--tool-call-parser", "qwen3_coder", \
     "--reasoning-parser", "qwen3", \
     "--attention-backend", "FLASHINFER", \
     "--speculative-config", "{\"method\":\"mtp\",\"num_speculative_tokens\":3}", \
     "--default-chat-template-kwargs", "{\"preserve_thinking\":false,\"enable_thinking\":false}", \
     "--compilation-config", "{\"cudagraph_mode\":\"PIECEWISE\"}", \
     "--no-use-tqdm-on-load", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
