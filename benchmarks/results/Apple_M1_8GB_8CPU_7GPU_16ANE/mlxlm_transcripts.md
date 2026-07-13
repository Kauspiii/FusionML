# MLX-LM TTFT proof transcripts — mlx-community/Qwen2.5-1.5B-Instruct-bf16
Greedy decoding; stock vs FusionML-patched runs on identical prompts.

## Prompt (1597 tokens; shown truncated)
```
...ared resources such as memory bandwidth make additional compute units irrelevant. The study of heterogeneous computing on unified memory architectures raises questions about when concurrent execution across processing units yields real speedups and when shared resources such as memory bandwidth make additional compute units irrelevant. Question: summarize the key trade-off in one sentence. Answer:
```

**Stock output:**

> The key trade-off in heterogeneous computing on unified memory architectures is between the benefits of concurrent execution across processing units and the limitations imposed by shared resources such as memory bandwidth. The key trade-off in heterogeneous computing on unified memory architectures is between the benefits of concurrent execution across processing units and the limitations imposed by shared resources such as memory

**FusionML output:**

> The key trade-off in heterogeneous computing on unified memory architectures is between the benefits of concurrent execution across processing units and the limitations imposed by shared resources such as memory bandwidth. The key trade-off in heterogeneous computing on unified memory architectures is between the benefits of concurrent execution across processing units and the limitations imposed by shared resources such as memory

**Token-identical: True**
