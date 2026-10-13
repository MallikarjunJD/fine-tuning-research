# Parameter-Efficient Fine-Tuning of Qwen2.5-1.5B for Agricultural Q&A

A controlled study of **LoRA vs QLoRA** for adapting a small instruction-tuned LLM to agricultural question answering — with a focus on **task performance, response behavior, computational efficiency, and failure analysis**.

The project uses **Qwen2.5-1.5B-Instruct** and the `KisanVaani/agriculture-qa-english-only` dataset to investigate what parameter-efficient fine-tuning actually improves, what it does not solve, and why retrieval becomes important for reliable agricultural AI.

---

## Links

- GitHub Repository: https://github.com/MallikarjunJD/fine-tuning-research
- LoRA Adapter: https://huggingface.co/MallikarjunJadi/qwen2.5-1.5b-agri-qa-lora
- QLoRA Adapter: https://huggingface.co/MallikarjunJadi/qwen2.5-1.5b-agri-qa-qlora
- Detailed Report: PROJECT_REPORT.md 


## Key Results

| Configuration | ROUGE-L | BERTScore F1 | Token F1 |
|---|---:|---:|---:|
| Base | 0.1074 | 0.8482 | 0.1590 |
| **LoRA** | **0.3559** | **0.9046** | **0.3973** |
| QLoRA | 0.2189 | 0.8784 | 0.2649 |

**Key result:** LoRA achieved the strongest overall performance, while QLoRA still provided a substantial improvement over the base model with the memory advantages of 4-bit quantization.

The experiments also revealed an important ML engineering lesson: an evaluation-pipeline bug initially made the models appear nearly identical. Rebuilding the evaluation framework exposed the actual differences between Base, LoRA, and QLoRA.

---

## Why This Project?

General-purpose instruction-tuned LLMs can answer agricultural questions, but their responses may be:

- Too generic
- Excessively verbose
- Region-inappropriate
- Factually unsupported
- Inconsistent with the concise style of curated agricultural QA

This project asks:

> **Can a small LLM be adapted to agricultural Q&A using parameter-efficient fine-tuning, and what are the trade-offs between LoRA and QLoRA?**

More importantly:

> **Does fine-tuning actually solve the knowledge problem — or does it mainly improve model behavior?**

---

## Research Questions

The study focuses on four questions:

1. Does LoRA/QLoRA improve agricultural QA compared with the untouched base model?
2. How does QLoRA compare with conventional LoRA in downstream performance?
3. How does fine-tuning change response behavior, particularly answer length and style?
4. What failure modes remain after fine-tuning, and which limitations suggest the need for RAG?

---

## What I Built

A complete fine-tuning and evaluation pipeline:

```text
Agricultural QA Dataset
        ↓
Cleaning & Preprocessing
        ↓
Leakage-Safe Train / Validation / Test Split
        ↓
Prompt Formatting
        ↓
Qwen2.5-1.5B-Instruct
        │
        ├───────────────┐
        ↓               ↓
      LoRA            QLoRA
      BF16          4-bit NF4
        │               │
        └───────┬───────┘
                ↓
        Unified Evaluation
                ↓
     ┌──────────┼──────────┐
     ↓          ↓          ↓
 ROUGE-L    BERTScore    Token F1
                ↓
       Qualitative Analysis
                ↓
       Failure Analysis
```

The repository contains both reusable scripts and notebooks documenting the experiments.

---

## Dataset

**[KisanVaani/agriculture-qa-english-only](https://huggingface.co/datasets/KisanVaani/agriculture-qa-english-only)**

The dataset contains agricultural question-answer pairs covering areas such as:

- Crop production
- Soil management
- Animal husbandry
- General farming practices

### Experimental Subset

| Split | Samples |
|---|---:|
| Training | 4,000 |
| Validation | 500 |
| Test | 500 |
| Qualitative reserve | 20 |

Preprocessing included:

- Removing empty/malformed answers
- Whitespace normalization
- Removing question-echoing answers where applicable
- Deterministic shuffling
- Leakage-safe splitting

The 4,000-example training subset was intentionally chosen to keep experimentation practical within the available compute budget.

---

## Model

### Base Model

**[Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)**

It was selected because it is:

- Small enough for practical experimentation on free Kaggle GPUs
- Instruction-tuned
- Compatible with the Hugging Face PEFT ecosystem
- Large enough to demonstrate meaningful LoRA/QLoRA behavior
- Suitable for future integration into an agricultural RAG system

---

## LoRA vs QLoRA

### LoRA

The LoRA experiment uses:

- Rank `r = 16`
- Attention projection adaptation
- Frozen base weights
- BF16 base-model training

Instead of updating the entire model, only low-rank adapter parameters are trained.

### QLoRA

The QLoRA experiment uses:

- 4-bit NF4 quantization
- Double quantization
- LoRA adapters
- Frozen quantized base weights

The purpose was to evaluate the practical trade-off between **memory efficiency and task performance**.

---

## Results

### Quantitative Performance

```text
ROUGE-L

Base     ███████
QLoRA    ███████████████
LoRA     ████████████████████████
```

| Model | ROUGE-L | BERTScore F1 | Token F1 |
|---|---:|---:|---:|
| Base | 0.1074 | 0.8482 | 0.1590 |
| **LoRA** | **0.3559** | **0.9046** | **0.3973** |
| QLoRA | 0.2189 | 0.8784 | 0.2649 |

### Response Behavior

Fine-tuning also changed the model's response length:

| Model | Average Words |
|---|---:|
| Base | 94.67 |
| QLoRA | 35.38 |
| **LoRA** | **24.15** |
| Reference Answers | 23.62 |

The base model produced considerably more verbose responses, while LoRA adapted closely to the response-length distribution of the target dataset.

### What the Results Mean

**LoRA**
- Best overall performance
- Strongest alignment with the target QA dataset
- Strong change in response style and conciseness

**QLoRA**
- Significant improvement over the base model
- Lower performance than LoRA in this experiment
- Provides the practical memory benefits of quantized fine-tuning

**Base**
- Useful baseline
- More verbose responses
- Lower task-specific alignment

---

## A Critical Evaluation Lesson

One of the most valuable discoveries was not a model result.

During the initial evaluation, Base, LoRA, and QLoRA frequently produced nearly identical outputs.

The first assumption was that fine-tuning had failed.

The actual problem was in the **evaluation pipeline**: adapters were being loaded onto the same underlying model object, causing contamination between evaluation modes.

The evaluation framework was redesigned with:

- Independent adapter isolation
- Model self-tests
- Adapter integrity verification
- Unified inference settings
- Identical evaluation data and prompts

After the redesign, the models produced meaningful differences and the quantitative results above became visible.

> **A model can train correctly and still appear broken if the evaluation pipeline is wrong.**

---

## Fine-Tuning Does Not Guarantee Correctness

The fine-tuned models improved substantially, but qualitative analysis revealed that they could still produce plausible yet incorrect agricultural facts.

Examples included:

- Incorrect cassava variety names
- Incorrect disease vectors
- Confusion between related agricultural diseases

This revealed an important distinction:

```text
Successful Training
        ≠
Correct Evaluation

Correct Evaluation
        ≠
Factual Accuracy

Fine-Tuning
        ↓
Better task behavior
Better response style
Better dataset alignment
        ≠
New verified knowledge

Retrieval (RAG)
        ↓
Grounded facts
Source-backed answers
```

Fine-tuning helps the model learn **how to answer the task**.

It does not automatically provide reliable, up-to-date external knowledge.

That limitation motivates the next stage of the project: **Fine-Tuning + RAG**.

---

## Hardware

| Hardware | Usage |
|---|---|
| NVIDIA T4 / P100 | Primary training |
| RTX 2050 4 GB | Local inference experiments |

Training was performed on Kaggle GPU infrastructure.

The local RTX 2050 was used primarily for inference experimentation because the available 4 GB VRAM was insufficient for the full training workload.

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- TRL / `SFTTrainer`
- bitsandbytes
- Evaluate
- Kaggle GPU
- Hugging Face Hub

---

## Evaluation

The project uses multiple complementary evaluation signals:

### Automatic

- **ROUGE-L** — lexical overlap
- **BERTScore F1** — semantic similarity
- **Token F1** — token-level overlap

### Behavioral

- Response-length analysis
- Qualitative comparison
- Failure-case analysis

Automatic metrics are treated as indicators rather than complete measures of agricultural correctness.

---

## Reproduction

### Clone

```bash
git clone https://github.com/MallikarjunJD/fine-tuning-research.git
cd fine-tuning-research
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train LoRA

```bash
python src/train.py --mode lora --output_dir outputs/lora
```

### Train QLoRA

```bash
python src/train.py --mode qlora --output_dir outputs/qlora
```

### Evaluate

```bash
# Base model
python src/evaluate.py \
    --output_json results/baseline_eval.json

# LoRA
python src/evaluate.py \
    --adapter_path outputs/lora \
    --output_json results/lora_eval.json

# QLoRA
python src/evaluate.py \
    --adapter_path outputs/qlora \
    --output_json results/qlora_eval.json
```

### Inference

```bash
python src/inference.py \
    --adapter_path outputs/qlora \
    --question "How do I control aphids on cotton?"
```

---

## Kaggle

The complete training experiments can be run using the notebooks provided in `notebooks/`.

The training pipeline requires:

- Kaggle GPU accelerator
- Internet access enabled
- Hugging Face model/dataset access

The experiments were designed around a 4,000-example training subset to keep runtime practical.

---

## Hugging Face Adapters

The trained adapters are available on Hugging Face:

- **[LoRA Adapter](https://huggingface.co/MallikarjunJadi/qwen2.5-1.5b-agri-qa-lora)**
- **[QLoRA Adapter](https://huggingface.co/MallikarjunJadi/qwen2.5-1.5b-agri-qa-qlora)**

Only the adapters are hosted; the base model is loaded separately.

---

## Demo

A temporary Gradio interface was used during evaluation to compare:

- Base Model
- LoRA
- QLoRA

The interface was primarily used for qualitative comparison and screen-recorded demonstrations during the experiment.

---

## Project Structure

```text
fine-tuning-research/
│
├── README.md
├── PROJECT_REPORT.md
│  
├── tests/
│   └── test_agri_llm.py
│  
├── images/
│   ├── Metrics.png
│   ├── PassedSelfTest.png
│   ├── Response1.png
│   ├── Response2.png
│   └── Response3.png
│
├── notebooks/
│   ├── Lora&Qlora_Training.ipynb
│   └── Inference, evaluation & deployment(LoRA&QLoRA).ipynb
│
├── src/
│   ├── agri_llm.py
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── scripts/
│   └── push_adapter_to_hub.py
│
├── app/
│   ├── app.py
│   ├── app_gradio.py
│   └── requirements.txt
│
├── configs/
│   └── training_config.yaml
│
├── results/
│   ├── baseline_results.md
│   └── fine_tuned_results.md
│
├── requirements.txt
└── .gitignore
```

---

## Detailed Report

A full technical report covering the experimental methodology, evaluation framework, quantitative results, failure analysis, limitations, and future work is available in:

**[PROJECT_REPORT.md](PROJECT_REPORT.md)**

---
## Screenshots

### Evaluation Validation
![Metrics](images/Metrics.png)

### Quantitative Results
images/metrics.png

### Base vs LoRA vs QLoRA
images/comparison1.png


## Limitations

This experiment has several important limitations:

- The dataset is community-curated rather than expert-verified.
- Only 4,000 training examples were used.
- ROUGE-L, BERTScore, and Token F1 are not substitutes for expert agronomic evaluation.
- The current system does not implement RAG.
- Fine-tuning does not guarantee factual correctness.
- The planned LoRA-rank ablation (`r ∈ {4, 16, 64}`) was not completed within the available compute/time budget.

These limitations are discussed in detail in the **[Project Report](PROJECT_REPORT.md)**.

---

## Future Work

The next stage is an:

### Agricultural AI Advisory System using Fine-Tuning + RAG

The planned system will combine the behavioral adaptation learned through fine-tuning with retrieval from verified agricultural documents.

```text
User Question
      ↓
   Retriever
      ↓
Relevant Agricultural Documents
      ↓
Retrieved Facts
      ↓
Fine-Tuned Model
      ↓
Grounded Answer
      ↓
Source Attribution
```

Planned components include:

- Agricultural document corpus
- FAISS vector search
- Retrieval pipeline
- Grounded generation
- Source attribution
- Fine-Tuned + RAG evaluation

The goal is to address the central limitation identified in this project:

> **Fine-tuning can improve how a model answers without guaranteeing that the facts it provides are correct.**

---

## Major Takeaway

This project started as a comparison between **LoRA and QLoRA**.

It ended up teaching a much broader ML engineering lesson:

> **Good ML experimentation requires both a good model pipeline and a trustworthy evaluation pipeline.**

LoRA demonstrated the strongest task performance in this experiment.

QLoRA demonstrated that quantized fine-tuning can still provide meaningful improvements under constrained memory.

And the failure analysis demonstrated why fine-tuning alone is not enough for a knowledge-intensive agricultural advisory system.

That is the motivation for the next project: **Fine-Tuning + RAG**.

---

## License

MIT for the code in this repository.

The base model and dataset have their own licenses and usage conditions. Refer to their respective Hugging Face pages before downstream or commercial use.

---

## Author

**Mallikarjun Jadi**  
Machine Learning Engineer

This project is part of a broader research and engineering journey focused on building trustworthy AI systems for agriculture through LLM fine-tuning, retrieval-augmented generation, and evaluation-driven development.

GitHub: https://github.com/MallikarjunJD  
Hugging Face: https://huggingface.co/MallikarjunJadi
Linkedin: https://www.linkedin.com/in/mallikarjunjadi
