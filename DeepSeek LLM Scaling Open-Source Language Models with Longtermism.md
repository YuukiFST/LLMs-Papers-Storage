# DeepSeek LLM: Scaling Open-Source Language Models with Longtermism

## Executive Summary

DeepSeek LLM represents a systematic investigation into scaling laws for large language models, introducing a 7B and 67B parameter model family trained on 2 trillion tokens. The research establishes novel scaling law formulations using non-embedding FLOPs/token as the model scale representation, replacing conventional parameter counts. Core findings demonstrate that optimal model/data allocation strategy exhibits data quality dependence: higher quality data necessitates increased compute allocation toward model scaling rather than data scaling. The 67B model surpasses LLaMA-2 70B across multiple benchmarks, achieving 18.7% on MATH (vs. 13.5%), 63.4% on GSM8K (vs. 58.4%), and 42.7% on HumanEval (vs. 28.7%). Post-alignment, DeepSeek 67B Chat achieves performance comparable to GPT-3.5 on open-ended evaluations while maintaining superior safety scores (97.8 on Do-Not-Answer benchmark).

---

## 1. Technical Architecture

### 1.1 Motivation and Problem Statement

Previous scaling law research presented inconsistent conclusions regarding optimal model/data allocation strategies when increasing compute budgets. The discrepancies between Kaplan et al. (2020) and Hoffmann et al. (2022) created uncertainty about generalizable scaling principles. Additionally, existing works inadequately addressed hyperparameter selection across different compute budgets, leaving open questions about whether models achieved optimal performance. The project addresses these uncertainties while developing open-source models in 7B and 67B configurations for the Chinese and English language domains.

### 1.2 Architecture Components

DeepSeek LLM adopts the LLaMA architecture foundation with specific modifications. The architecture employs Pre-Norm structure with RMSNorm normalization, SwiGLU activation functions in Feed-Forward Networks with intermediate dimension of (8/3)d_model, and Rotary Position Embeddings (RoPE) for positional encoding.

Model specifications diverge from standard configurations in layer allocation. DeepSeek 7B implements 30 layers with 4096 model dimension, 32 attention heads, and 32 key-value heads. DeepSeek 67B implements 95 layers with 8192 model dimension, 64 attention heads, and 8 key-value heads using Grouped-Query Attention (GQA). The 67B model expands parameters through network depth rather than FFN width to optimize performance.

Vocabulary configuration uses Byte-level Byte-Pair Encoding (BBPE) with 100,000 conventional tokens plus 15 special tokens, extended to 102,400 for computational efficiency and future extensibility. The tokenizer prevents merging across character categories (newlines, punctuation, CJK symbols) and splits numbers into individual digits.

Mathematical formulation for non-embedding FLOPs/token:

```
M = 72 * n_layers * d_model^2 + 12 * n_layers * d_model * l_seq
```

where n_layers is layer count, d_model is model width, and l_seq is sequence length (4096). This metric accounts for attention computational overhead while excluding vocabulary computation.

### 1.3 Training or Pre-Training Protocol

Pre-training utilizes 2 trillion tokens from multilingual sources with emphasis on Chinese and English. Data pipeline implements three-stage processing: deduplication, filtering, and remixing. Aggressive deduplication across 91 Common Crawl dumps achieves 89.8% deduplication rate compared to 22.2% for single-dump methods.

Initialization employs standard deviation 0.006. Optimization uses AdamW with β1=0.9, β2=0.95, weight_decay=0.1, and gradient clipping at 1.0. Training precision is bf16 for forward/backward passes with fp32 gradient accumulation.

Multi-step learning rate scheduler replaces cosine scheduling. Learning rate reaches maximum after 2000 warmup steps, decreases to 31.6% of maximum after 80% of tokens, and further reduces to 10% after 90% of tokens. This design facilitates continual training through phase reuse while maintaining performance equivalence with cosine scheduling.

Model-specific hyperparameters derived from scaling laws:
- 7B: batch size 2304, learning rate 4.2e-4, 2.0T tokens
- 67B: batch size 4608, learning rate 3.2e-4, 2.0T tokens

Checkpoint saving occurs asynchronously every 5 minutes, limiting maximum training loss to 5 minutes in failure scenarios.

### 1.4 Performance Impact

Infrastructure employs HAI-LLM framework integrating data parallelism, tensor parallelism, sequence parallelism, and 1F1B pipeline parallelism with ZeRO-1 optimizer state partitioning. Flash attention optimizes hardware utilization. Computation-communication overlap minimizes waiting overhead including backward pass/reduce-scatter overlap and GEMM/all-gather/reduce-scatter overlap.

Performance scaling predictions demonstrate accuracy across 1000× compute budget range. Validation set bits-per-byte for 7B and 67B models align precisely with extrapolated scaling curves from smaller-scale experiments (1e17 to 3e20 FLOPs).

Efficiency optimizations include layer/operator fusion (LayerNorm, GEMM, Adam updates) and in-place cross-entropy computation converting bf16 logits to fp32 in CUDA kernel execution rather than pre-conversion in HBM, reducing memory consumption.

---

## 2. Post-Training or Optimization Methods

Alignment pipeline implements two-stage methodology: Supervised Fine-Tuning (SFT) followed by Direct Preference Optimization (DPO).

SFT utilizes 1.5 million instruction instances: 1.2 million helpful data (31.2% general language, 46.6% mathematics, 22.2% coding) and 300K safety data across sensitive topics. Training duration is 4 epochs for 7B model and 2 epochs for 67B model to mitigate overfitting. Learning rates are 1e-5 (7B) and 5e-6 (67B).

Two-stage fine-tuning addresses repetition issues in smaller models. Stage 1 uses all data; Stage 2 excludes math/code data to reduce repetition ratio from 2.0% to 1.4% in 7B model while maintaining benchmark performance. The 67B model exhibits sub-1% repetition after single-stage SFT.

DPO constructs preference data for helpfulness and harmlessness dimensions. Training executes one epoch with learning rate 5e-6, batch size 512, warmup and cosine scheduling. DPO enhances open-ended generation capabilities with minimal impact on standard benchmark performance.

Key findings: Math and code performance improvements (HumanEval +22 points, GSM8K +20+ points for 7B) attributed to base model underfitting on these domains. SFT learns reasoning format rather than reasoning capability itself. Performance degradation observed on cloze/completion tasks (HellaSwag) where language modeling objectives better align with evaluation format.

---

## 3. Agentic or System-Level Design (if applicable)

Not applicable per source document.

---

## 4. Benchmark Performance and Ablations

### Base Model Performance

| Benchmark | LLaMA2 7B | DeepSeek 7B | LLaMA2 70B | DeepSeek 67B |
|-----------|-----------|-------------|------------|--------------|
| MMLU (5-shot) | 45.8 | 48.2 | 69.0 | 71.3 |
| GSM8K (8-shot) | 15.5 | 17.4 | 58.4 | 63.4 |
| MATH (4-shot) | 2.5 | 6.0 | 13.5 | 18.7 |
| HumanEval (0-shot) | 14.6 | 26.2 | 28.7 | 42.7 |
| MBPP (3-shot) | 21.8 | 39.0 | 45.6 | 57.4 |
| BBH (3-shot) | 38.5 | 39.5 | 62.9 | 68.7 |
| C-Eval (5-shot) | 33.9 | 45.0 | 51.4 | 66.1 |
| CMMLU (5-shot) | 32.6 | 47.2 | 53.1 | 70.8 |

### Chat Model Performance

| Benchmark | DeepSeek 7B Base | DeepSeek 7B Chat | DeepSeek 67B Base | DeepSeek 67B Chat |
|-----------|------------------|------------------|-------------------|-------------------|
| GSM8K | 17.4 | 63.0 | 63.4 | 84.1 |
| MATH | 6.0 | 15.8 | 18.7 | 32.6 |
| HumanEval | 26.2 | 48.2 | 42.7 | 73.8 |
| MBPP | 39.0 | 35.2 | 57.4 | 61.4 |
| BBH | 39.5 | 42.3 | 68.7 | 71.7 |

### Open-Ended Evaluation

MT-Bench scores (English):
- GPT-4-1106-preview: 9.26
- DeepSeek 67B Chat DPO: 8.76
- DeepSeek 67B Chat: 8.35
- GPT-3.5-turbo-0613: 8.39
- LLaMA-2-Chat 70B: 6.86

AlignBench scores (Chinese):
- GPT-4-1106-preview: 8.01
- DeepSeek 67B Chat DPO: 6.69
- DeepSeek 67B Chat: 6.43
- GPT-3.5-turbo-0613: 6.08

### Held-Out Dataset Performance

| Model | LeetCode | Hungarian Exam | IFEval |
|-------|----------|----------------|--------|
| GPT-4 | 48.4 | 68 | 79.3 |
| DeepSeek 67B Chat | 17.5 | 58 | 55.5 |
| Yi-Chat 34B | 7.9 | 39 | 48.4 |
| DeepSeek 7B Chat | 4.7 | 28.5 | 41.2 |

### Safety Evaluation

Do-Not-Answer safety scores:
- DeepSeek 67B Chat: 97.8
- ChatGPT: 97.7
- GPT-4: 96.5

Internal safety taxonomy evaluation: 2400 questions across 5 categories with 2097/2400 safe responses.

### Scaling Law Ablations

Optimal hyperparameter scaling with compute budget C:

```
η_opt = 0.3118 * C^(-0.1250)
B_opt = 0.2920 * C^(0.3271)
```

Optimal model/data allocation:

```
M_opt = 0.1715 * C^(0.5243)
D_opt = 5.8316 * C^(0.4757)
```

Data quality impact on scaling exponents:

| Dataset | Model Scaling (a) | Data Scaling (b) |
|---------|-------------------|------------------|
| Early In-house | 0.450 | 0.550 |
| Current In-house | 0.524 | 0.476 |
| OpenWebText2 | 0.578 | 0.422 |
| Chinchilla (MassiveText) | 0.49 | 0.51 |
| OpenAI (OpenWebText2) | 0.73 | 0.27 |

---

## 5. Key Technical Takeaways

- Non-embedding FLOPs/token (M) provides more accurate model scale representation than parameter counts, reducing approximation error from 50% to near-zero for small models
- Optimal batch size increases and learning rate decreases as power laws of compute budget, with near-optimal parameters occupying broad band ranges
- Data quality significantly influences optimal model/data scaling allocation; higher quality data necessitates increased model scaling allocation (higher exponent a, lower exponent b)
- Multi-step learning rate scheduler achieves equivalent performance to cosine scheduler while enabling continual training through phase reuse
- Small-scale experiments (1e17 FLOPs) accurately predict performance at 1000× compute scale (3e20 FLOPs)
- Performance advantage of 67B over LLaMA-2 70B exceeds that of 7B over LLaMA-2 7B, indicating greater language conflict impact on smaller models
- Two-stage SFT mitigates repetition behavior in smaller models without degrading benchmark performance
- DPO enhances open-ended generation with minimal standard benchmark impact
- Instruction data in pre-training provides no advantage over equivalent SFT-stage inclusion
- System prompts benefit large models (67B) but degrade small model (7B) performance
- Multi-choice question data improves MC benchmark scores but does not transfer to generative evaluation formats

---

## 6. Conclusion

DeepSeek LLM establishes refined scaling law methodology through non-embedding FLOPs/token representation and demonstrates data quality dependence in optimal model/data allocation strategies. The 7B and 67B models, trained on 2 trillion bilingual tokens with scaling law guidance, achieve competitive performance against larger models while maintaining open-source accessibility. The 67B Chat variant matches GPT-3.5 on open-ended evaluations and exceeds both GPT-3.5 and GPT-4 on safety benchmarks. Limitations include static knowledge cutoff, hallucination tendencies, incomplete Chinese data coverage in initial version, and limited multilingual capability beyond Chinese-English. Future work includes expanded code intelligence capabilities, Mixture-of-Experts architectures, enhanced reasoning through reinforcement learning, and dataset expansion for improved mathematical and Chinese knowledge coverage.

---

## References

- Paper: arXiv:2401.02954v1 [cs.CL] 5 Jan 2024
- Authors: 95 contributors from DeepSeek-AI (alphabetically ordered)
- Training: 2 trillion tokens, bilingual corpus (Chinese/English emphasis), multi-step learning rate scheduler, bf16 precision with fp32 gradient accumulation
- Evaluation: Internal framework spanning 30+ benchmarks including MMLU, GSM8K, MATH, HumanEval, MBPP, BBH, C-Eval, CMMLU, MT-Bench, AlignBench
- Architecture references: LLaMA (Touvron et al., 2023a,b), Transformer (Vaswani et al., 2017), GQA (Ainslie et al., 2023), RoPE (Su et al., 2024), SwiGLU (Shazeer, 2020), RMSNorm (Zhang and Sennrich, 2019)
- Scaling laws: Kaplan et al. (2020), Hoffmann et al. (2022), Henighan et al. (2020)
- Alignment: SFT methodology (Ouyang et al., 2022), DPO (Rafailov et al., 2023)
- Infrastructure: HAI-LLM (High-flyer, 2023), Megatron components (Korthikanti et al., 2023; Narayanan et al., 2021; Shoeybi et al., 2019), FlashAttention (Dao, 2023; Dao et al., 2022), ZeRO (Rajbhandari et al., 2020), vLLM (Kwon et al., 2023)
- Tokenizer: Byte-level BPE via Huggingface tokenizers library (2019), 100,015 vocabulary size extended to 102,400
- Data sources: Common Crawl deduplication approach informed by RedPajama (Computer, 2023), The Pile (Gao et al., 2020), RefinedWeb (Penedo et al., 2023)
