# Parameter-Efficient Fine-Tuning of Qwen2.5-1.5B for Agricultural Question Answering

## Abstract

This project investigates whether a small instruction-tuned language model can be adapted to agricultural question answering using Parameter-Efficient Fine-Tuning (PEFT), while remaining practical under limited computational resources.

The study compares three configurations of **Qwen2.5-1.5B-Instruct**:

1. The original base model without fine-tuning
2. LoRA fine-tuning using a BF16 base model
3. QLoRA fine-tuning using a 4-bit NF4-quantized base model

The experiments were conducted using a controlled subset of the **KisanVaani Agriculture QA** dataset consisting of 4,000 training examples, 500 validation examples, and 500 test examples, with an additional 20 examples reserved for qualitative comparison.

The models were evaluated using ROUGE-L, BERTScore F1, Token F1, response-length analysis, and qualitative inspection. LoRA achieved the strongest overall performance, increasing ROUGE-L from **0.1074 for the base model to 0.3559**, while BERTScore F1 increased from **0.8482 to 0.9046**.

QLoRA also substantially improved over the base model, although it underperformed LoRA on the measured task metrics.

An important engineering discovery occurred during evaluation: an adapter-loading issue caused different model configurations to produce nearly identical outputs. The evaluation framework was subsequently redesigned around adapter isolation, model self-tests, adapter-integrity checks, and a unified inference pipeline.

The project demonstrates that PEFT can substantially improve task-specific behavior in a small language model. However, the qualitative failure analysis also shows that fine-tuning does not guarantee factual correctness. The resulting limitations motivate the integration of **Retrieval-Augmented Generation (RAG)** in the next stage of the Agricultural AI Advisory System.

---

# 1. Introduction

Large language models can answer questions across a wide range of domains, including agriculture. However, a general-purpose instruction-tuned model may produce answers that are too generic, unnecessarily verbose, region-inappropriate, or insufficiently grounded in agricultural knowledge.

Agricultural question answering presents an especially important challenge because users may ask highly specific questions about crops, diseases, soil management, pests, varieties, and farming practices.

This project investigates whether a relatively small language model can be adapted to this domain using **Parameter-Efficient Fine-Tuning (PEFT)** rather than updating all model parameters.

The central focus is not simply obtaining the highest evaluation score. Instead, the project attempts to understand:

- What changes when a small LLM is fine-tuned on agricultural QA?
- How does LoRA compare with QLoRA?
- How much improvement can be obtained using a relatively small training subset?
- What types of errors remain after fine-tuning?
- Which limitations belong to fine-tuning itself and which are better addressed through retrieval?

This work also serves as the fine-tuning component of a larger planned **Agricultural AI Advisory System using Fine-Tuning + RAG**.

---

# 2. Problem Statement

A general-purpose instruction-tuned LLM can generate agricultural answers, but its responses may not consistently match the concise and specific style of a curated agricultural QA dataset.

Common problems include:

- Generic responses
- Excessive verbosity
- Incorrect or unsupported agricultural facts
- Region-inappropriate recommendations
- Confusion between related agricultural concepts

The problem addressed by this project is therefore:

> **Can Parameter-Efficient Fine-Tuning adapt a small instruction-tuned LLM to agricultural question answering while remaining computationally practical on limited hardware?**

The project further investigates whether the efficiency advantages of QLoRA come with a measurable reduction in task performance compared with conventional LoRA.

---

# 3. Research Objectives

The project was designed around four primary objectives.

### Objective 1 — Task Adaptation

Determine whether fine-tuning Qwen2.5-1.5B-Instruct improves agricultural question-answering performance compared with the untouched base model.

### Objective 2 — LoRA vs QLoRA

Compare LoRA and QLoRA in terms of task performance while considering their different memory requirements.

### Objective 3 — Behavioral Analysis

Determine how fine-tuning changes the model's response characteristics, including response length, style, structure, and specificity.

### Objective 4 — Failure Analysis

Identify the types of factual errors that remain after fine-tuning and determine why fine-tuning alone may not be sufficient for reliable agricultural knowledge delivery.

---

# 4. Research Questions

The study is centered around the following questions:

### RQ1
Does LoRA/QLoRA fine-tuning improve agricultural question-answering performance compared with the untouched base model?

### RQ2
How does QLoRA compare with conventional LoRA in terms of downstream answer quality?

### RQ3
How does fine-tuning change the behavioral characteristics of the model, particularly response length and answer style?

### RQ4
What failure modes remain after fine-tuning, and which of those limitations suggest the need for retrieval-based grounding?

---

# 5. Dataset

## 5.1 Dataset Source

The experiments use:

**KisanVaani/agriculture-qa-english-only**

The dataset contains English agricultural question-answer pairs covering areas such as:

- Crop production
- Soil management
- Animal husbandry
- General farming practices

The original dataset contains approximately 22.6k examples.

The dataset is community-curated and aggregated from agricultural forums and FAQ sources. Consequently, its answers should be treated as a research dataset rather than as authoritative agronomic ground truth.

---

## 5.2 Experimental Subset

To keep the experiments computationally practical, a controlled subset was used:

| Split | Samples |
|---|---:|
| Training | 4,000 |
| Validation | 500 |
| Test | 500 |
| Qualitative reserve | 20 |

The 20 qualitative examples were held aside for manual side-by-side comparison.

---

## 5.3 Preprocessing

The preprocessing pipeline included:

1. Removal of empty answers
2. Removal of malformed records
3. Whitespace normalization
4. Removal of question-echoing answers where applicable
5. Deterministic shuffling
6. Leakage-safe dataset splitting

The train, validation, and test split was performed before downstream sampling to prevent information leakage between experimental subsets.

---

# 6. Base Model

## 6.1 Qwen2.5-1.5B-Instruct

The selected base model was:

**Qwen/Qwen2.5-1.5B-Instruct**

The model was selected because it provides a practical balance between model capability and computational requirements.

The selection was motivated by:

- Instruction-following capability
- Small parameter count relative to larger contemporary LLMs
- Compatibility with the Hugging Face PEFT ecosystem
- Ability to train using freely available Kaggle GPU resources
- Suitability for controlled LoRA and QLoRA experiments
- Potential integration into the larger Agricultural AI Advisory System

The base model was evaluated without fine-tuning to establish a controlled baseline.

---

# 7. Experimental Design

The experiment compares three model configurations.

```text
                    Qwen2.5-1.5B-Instruct
                             │
              ┌──────────────┼──────────────┐
              │              │              │
            Base            LoRA           QLoRA
              │              │              │
          No training    BF16 + LoRA   4-bit NF4 + LoRA
              │              │              │
              └──────────────┼──────────────┘
                             │
                    Common Evaluation
                             │
              ┌──────────────┼──────────────┐
              │              │              │
           ROUGE-L       BERTScore      Token F1
                             │
                     Qualitative Analysis
```

The same underlying evaluation dataset and prompt structure were used to make the comparison as consistent as possible.

---

# 8. LoRA Fine-Tuning

## 8.1 Approach

Low-Rank Adaptation (LoRA) was used to adapt the model without updating the complete set of base-model parameters.

The base model remained frozen while trainable low-rank adapter matrices were introduced into the attention projections.

### Configuration

- Base model: Qwen2.5-1.5B-Instruct
- Adapter rank: `r = 16`
- Target: attention projections
- Base precision: BF16
- Training method: supervised fine-tuning

The resulting adapter contains only a small fraction of the parameters of the original model.

---

# 9. QLoRA Fine-Tuning

QLoRA extends the LoRA approach by quantizing the base model while training the adapter.

The experimental configuration used:

- 4-bit base-model quantization
- NF4 quantization
- Double quantization
- LoRA adapters
- Frozen quantized base weights

The goal was to determine whether the memory savings provided by quantization could be achieved without unacceptable degradation in task performance.

The same general data, prompt structure, and effective training setup were maintained to make the LoRA and QLoRA comparison meaningful.

---

# 10. Training Infrastructure

## 10.1 Kaggle

Training was performed using Kaggle GPU infrastructure.

The available configurations included:

- NVIDIA T4
- NVIDIA P100
- Approximately 16 GB VRAM per GPU configuration used

The Kaggle environment provided sufficient memory for the LoRA and QLoRA training experiments.

---

## 10.2 Local Hardware

The local development machine contained:

- NVIDIA RTX 2050
- 4 GB VRAM

The local GPU was not used for the primary training experiments.

Instead, it was used for post-training inference experiments and for understanding the practical memory requirements of running the model locally.

Although a 1.5B model quantized to 4-bit can have approximately 1 GB of weight storage, model execution requires additional memory for activations, KV cache, CUDA runtime overhead, and other processes.

This provided a practical demonstration that parameter count or raw weight size alone does not determine whether a model can be trained on a particular GPU.

---

# 11. Evaluation Methodology

The evaluation framework was designed to compare the three model configurations under the same conditions.

The primary evaluation metrics were:

- ROUGE-L
- BERTScore F1
- Token F1

In addition, response length and qualitative behavior were analyzed.

## 11.1 ROUGE-L

ROUGE-L was used to measure the similarity between generated answers and reference answers based on longest common subsequence relationships.

It provides a useful lexical-overlap signal, although it should not be interpreted as a complete measure of factual correctness.

## 11.2 BERTScore

BERTScore evaluates semantic similarity between generated and reference text using contextual representations.

It is useful for open-ended answers where exact wording may differ despite semantic similarity.

## 11.3 Token F1

Token-level F1 was included as an additional diagnostic measure of overlap between generated and reference answers.

## 11.4 Qualitative Evaluation

Automatic metrics were supplemented with manual inspection.

The qualitative evaluation examined:

- Correctness
- Specificity
- Response style
- Conciseness
- Plausibility
- Failure patterns

This was particularly important because a generated answer can have high semantic similarity while still containing an incorrect agricultural fact.

---

# 12. Evaluation Pipeline Failure

One of the most significant discoveries in the project occurred during evaluation.

Initially, the Base, LoRA, and QLoRA models frequently produced nearly identical outputs.

This created the appearance that:

- Fine-tuning had failed
- The adapters were ineffective
- The dataset was insufficient
- The models had learned very little

Rather than immediately assuming that training had failed, the evaluation pipeline itself was investigated.

---

# 13. Root Cause Analysis

The problem was traced to the way PEFT adapters were being loaded during evaluation.

Different adapter configurations were being evaluated on the same underlying model object, creating adapter contamination between evaluation modes.

This compromised the intended comparison:

```text
Expected:

Base  ────────────────→ Base output
LoRA  ────────────────→ LoRA output
QLoRA ────────────────→ QLoRA output


Problematic evaluation:

             Shared model object
              /       |       \
           Base      LoRA     QLoRA
              ↓        ↓        ↓
          Contaminated comparison
```

Consequently, the apparent similarity between model outputs could not be interpreted as evidence that fine-tuning had failed.

This became an important lesson in experimental ML engineering:

> **A valid training pipeline does not guarantee a valid evaluation pipeline.**

---

# 14. Evaluation Framework Redesign

Instead of applying a small patch to the existing notebook, the evaluation framework was redesigned to make model isolation and integrity explicit.

## 14.1 Adapter Isolation

Each evaluation mode was treated independently:

- Base
- LoRA
- QLoRA

The evaluation process was designed to prevent one adapter from affecting another model configuration.

---

## 14.2 Model Self-Test

Before running the full evaluation, model behavior was checked to ensure that different configurations were actually producing different internal outputs.

The self-test compared model logits and stopped evaluation if configurations appeared identical when they should differ.

The resulting self-test passed after the framework was redesigned.

---

## 14.3 Adapter Integrity Verification

The evaluation framework also verifies that:

- Adapter files exist
- Expected trained weights are present
- Adapter parameters are non-trivial

This reduces the risk of accidentally evaluating an untrained or incomplete adapter.

---

## 14.4 Unified Inference

All configurations use the same:

- Prompt format
- Evaluation dataset
- Decoding settings
- Evaluation procedure

This makes the final comparison substantially more controlled.

---

# 15. Quantitative Results

The final evaluation produced the following results:

| Model | ROUGE-L | BERTScore F1 | Token F1 |
|---|---:|---:|---:|
| Base | 0.1074 | 0.8482 | 0.1590 |
| **LoRA** | **0.3559** | **0.9046** | **0.3973** |
| QLoRA | 0.2189 | 0.8784 | 0.2649 |

---

# 16. Results Analysis

## 16.1 Base Model

The base model established the reference point for the experiment.

Its performance was:

- ROUGE-L: **0.1074**
- BERTScore F1: **0.8482**
- Token F1: **0.1590**

The relatively low ROUGE-L and Token F1 values indicate limited lexical alignment with the target answers.

The model was also considerably more verbose than the reference answers.

---

## 16.2 LoRA

LoRA produced the strongest performance across all three reported metrics.

Compared with the base model:

- ROUGE-L increased from **0.1074 → 0.3559**
- BERTScore F1 increased from **0.8482 → 0.9046**
- Token F1 increased from **0.1590 → 0.3973**

This indicates a substantial improvement in alignment with the agricultural QA dataset.

LoRA also produced a major behavioral change in response length.

---

## 16.3 QLoRA

QLoRA also improved substantially over the base model:

- ROUGE-L: **0.2189**
- BERTScore F1: **0.8784**
- Token F1: **0.2649**

However, it did not reach the performance of conventional LoRA in this experiment.

This demonstrates an important practical trade-off:

> QLoRA can provide the memory advantages of quantized fine-tuning while still improving task performance, but quantization may come with a measurable performance cost for a particular task and configuration.

---

# 17. Response-Length Analysis

Average generated response length was measured in words.

| Model | Average Words |
|---|---:|
| Base | 94.67 |
| QLoRA | 35.38 |
| **LoRA** | **24.15** |
| Reference Answers | 23.62 |

The base model generated substantially longer responses than the reference answers.

After fine-tuning, the response length moved much closer to the target distribution.

LoRA was particularly notable:

```text
Base       → 94.67 words
QLoRA      → 35.38 words
LoRA       → 24.15 words
Reference  → 23.62 words
```

This suggests that fine-tuning did not merely affect evaluation metrics. It also changed the model's response behavior toward the style represented in the training corpus.

---

# 18. Qualitative Failure Analysis

Quantitative improvement does not imply perfect factual reliability.

Several examples demonstrated that the fine-tuned models could still produce answers that sounded plausible while being factually incorrect.

---

## 18.1 Failure Case 1 — Incorrect Cassava Variety

### Question

> Which cassava variety was developed by IITA and is popular in many African countries?

### Reference Answer

**TME 419**

### LoRA Output

**NICO 102**

### QLoRA Output

**Kintan 1**

### Analysis

Both fine-tuned models produced plausible-sounding agricultural variety names but failed to identify the reference fact.

This is an important failure mode because the generated answers do not necessarily appear obviously nonsensical.

The problem is therefore not simply language generation quality; it is factual knowledge reliability.

---

# 19. Qualitative Failure Case 2 — Incorrect Disease Vector

### Question

> What spreads Cassava Mosaic Disease?

### Reference Answer

**Whitefly**

### Model Output

**Aphids**

### Analysis

The model produced another agriculturally plausible entity, but the specific fact was incorrect.

This illustrates how a model can understand the general domain while still retrieving the wrong fact from its learned representations.

---

# 20. Qualitative Failure Case 3 — Related but Incorrect Disease

### Question

> Which fungal disease affects cassava?

### Reference Answer

**Root rot**

### Model Output

**Cassava Mosaic Disease**

### Analysis

The generated response remained within the correct agricultural domain but failed to identify the requested disease.

The model therefore demonstrated domain familiarity without reliable fact-level retrieval.

---

# 21. What the Failure Analysis Shows

The qualitative examples reveal a recurring pattern:

> **Fine-tuning improved behavior, but it did not guarantee factual grounding.**

The model can learn:

- How answers should be structured
- How concise responses should be
- Which terminology appears frequently in the corpus
- What style is expected by the dataset

However, this does not guarantee that the model will retrieve the correct fact for every question.

This distinction is critical for the intended agricultural advisory application.

---

# 22. Fine-Tuning vs Retrieval

The results suggest two different roles for the two techniques.

### Fine-Tuning

Fine-tuning is useful for learning:

- Task behavior
- Response style
- Formatting
- Domain-specific conversational patterns
- Concise answer generation

### Retrieval-Augmented Generation

RAG can provide:

- External knowledge
- Updatable information
- Document-level grounding
- Source attribution
- Evidence for generated claims

The observed hallucination and fact-confusion failures therefore provide a practical motivation for adding retrieval rather than attempting to solve every knowledge problem through additional fine-tuning.

---

# 23. Ablation Study Status

A LoRA-rank ablation was initially planned using:

```text
r ∈ {4, 16, 64}
```

The purpose was to investigate the relationship between adapter capacity, trainable parameters, memory usage, and downstream performance.

However, the final experiment focused on the three primary configurations:

- Base
- LoRA
- QLoRA

The planned rank ablation was not completed within the available time and compute budget.

It is therefore treated as **future experimental work rather than a completed result**.

This distinction is maintained to avoid presenting planned experiments as measured findings.

---

# 24. Limitations

Several limitations should be considered when interpreting the results.

### 24.1 Dataset Quality

The dataset is community-curated and is not equivalent to expert-verified agronomic ground truth.

Its answers may contain:

- Regional biases
- Source-specific biases
- Outdated information
- Inconsistencies

---

### 24.2 Training Subset

Only 4,000 examples were used for training despite the considerably larger source dataset.

This was a deliberate computational constraint rather than a claim that 4,000 examples are sufficient for complete agricultural domain adaptation.

---

### 24.3 Automatic Evaluation

ROUGE-L, BERTScore, and Token F1 are useful indicators, but they are not substitutes for expert agricultural evaluation.

A response may achieve good semantic similarity while still containing a critical factual error.

---

### 24.4 No Retrieval Layer

The current project does not implement RAG.

Therefore, the fine-tuned model relies primarily on what it learned during training rather than retrieving verified information at inference time.

This limitation is intentional because this repository isolates and studies the fine-tuning component before integrating retrieval.

---

### 24.5 Limited Qualitative Evaluation

The qualitative analysis provides useful failure examples but should not be interpreted as a comprehensive agronomic safety evaluation.

A production agricultural advisory system would require substantially stronger validation and expert review.

---

# 25. Reproducibility

The project is organized as a reproducible experiment rather than a single training notebook.

The repository includes:

- Data-processing code
- Training scripts
- Evaluation scripts
- Inference scripts
- Training configuration
- Evaluation results
- Notebooks documenting the experiments

The primary pipeline follows:

```text
Dataset
   ↓
Cleaning
   ↓
Leakage-safe split
   ↓
Prompt formatting
   ↓
Base evaluation
   ↓
LoRA / QLoRA training
   ↓
Adapter verification
   ↓
Unified evaluation
   ↓
Quantitative analysis
   ↓
Qualitative analysis
   ↓
Failure analysis
```

Training and evaluation can be reproduced using the repository's scripts and Kaggle notebooks.

---

# 26. Project Artifacts

The project produces several reusable artifacts:

### LoRA Adapter

A trained LoRA adapter is hosted on Hugging Face.

### QLoRA Adapter

A trained QLoRA adapter is also hosted on Hugging Face.

### Evaluation Framework

The redesigned evaluation pipeline provides a safer framework for comparing base and adapter-based models.

### Experiment Documentation

The repository contains detailed experiment notes, results, and implementation documentation.

---

# 27. Future Work

The next stage of this research is an:

> **Agricultural AI Advisory System using Fine-Tuning + Retrieval-Augmented Generation**

The current project provides the fine-tuning component.

The next system will add a retrieval layer over agricultural documents.

Planned components include:

- Verified agricultural document corpus
- Document preprocessing
- Vector indexing
- FAISS-based retrieval
- Retrieval pipeline
- Grounded generation
- Source attribution
- Fine-tuned model integration
- Comparison between fine-tuning-only and fine-tuning + RAG

The conceptual architecture is:

```text
User Question
      ↓
   Retriever
      ↓
Relevant Agricultural Documents
      ↓
Retrieved Facts / Context
      ↓
Fine-Tuned Language Model
      ↓
Grounded Agricultural Answer
      ↓
Source Attribution
```

The goal is not to use RAG merely as another model component, but to address a specific limitation identified during this experiment: **fine-tuning can improve behavior without guaranteeing factual grounding.**

---

# 28. Conclusion

This project demonstrated that Parameter-Efficient Fine-Tuning can substantially improve the behavior of a small instruction-tuned language model on agricultural question answering.

Among the evaluated configurations, **LoRA achieved the strongest overall task performance**, substantially outperforming both the original base model and QLoRA on the reported evaluation metrics.

QLoRA also produced meaningful improvement over the base model while providing the practical memory advantages associated with 4-bit quantization.

However, the most important outcome was not simply the numerical improvement.

During the experiment, an evaluation-pipeline issue caused different model configurations to appear nearly identical. Investigating and resolving the problem demonstrated an important principle of machine learning experimentation:

> **Training a model correctly is only half of the experiment. The evaluation pipeline must also be correct.**

Once the evaluation framework was redesigned, meaningful differences between Base, LoRA, and QLoRA became visible.

The subsequent failure analysis revealed a second important lesson:

> **Fine-tuning improves task behavior, but it does not guarantee factual correctness.**

The model could produce concise, domain-appropriate, and plausible answers while still providing incorrect agricultural facts.

This directly motivates the next stage of the work: combining fine-tuning with retrieval-based grounding.

Therefore, the contribution of this project is not simply a fine-tuned agricultural LLM. It is a controlled investigation into **what PEFT improves, what it does not solve, how evaluation can fail, and why retrieval becomes necessary for a more reliable agricultural AI system.**