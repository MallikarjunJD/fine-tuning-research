# Fine-Tuned Results — LoRA vs QLoRA on Qwen2.5-1.5B-Instruct

Status: **[TO BE FILLED AFTER EXPERIMENT]**

## Quantitative comparison

| Metric | Base (no FT) |	LoRA |	QLoRA |
|---|---|---|---|
| ROUGE-L | 0.1074 | 0.3559 | 	0.2189 |
| BERTScore F1 | 0.8482 | 0.9046 | 0.8784 |
| Token F1 | 0.1590 | 0.3973 | 0.2649 |
| Exact match rate | 0.000 | 0.105 | 0.020 |
| Avg answer length | 94.67 |	24.15 |	35.38 |
| Reference length | 23.62 | 23.62 | 23.62 |

## LoRA Rank Ablation

Not performed in the final experiment.
Planned future work.

## Failure Analysis

Fine-tuning improved task-specific performance on the agricultural Q&A dataset, but several failure modes remained. Representative examples are shown below.

### 1. Hallucinated Agricultural Variety

**Question**

> Which cassava variety was developed by IITA and is popular in many African countries?

**Reference Answer**

> TME 419

**LoRA Output**

> NICO 102

**QLoRA Output**

> Kintan 1

**Analysis**

Both fine-tuned models generated plausible-looking cassava variety names that did not match the reference answer. This demonstrates a common limitation of fine-tuning: models may produce confident and domain-relevant responses without guaranteeing factual correctness.

---

### 2. Incorrect Pest Identification

**Question**

> What spreads Cassava Mosaic Disease (CMD)?

**Reference Answer**

> Whitefly

**Base Model Output**

> Aphids

**Analysis**

The base model generated an agriculturally plausible pest but failed to identify the correct vector. This illustrates how general-purpose language models can substitute related concepts when domain-specific knowledge is insufficient.

---

### 3. Relevant but Incorrect Disease Answer

**Question**

> Which fungal disease affects cassava?

**Reference Answer**

> Root rot

**LoRA Output**

> Cassava mosaic disease (CMD) and cassava brown streak disease (CBSD)

**Analysis**

The model demonstrated knowledge of common cassava diseases but failed to answer the specific question being asked. Instead of identifying the fungal disease referenced by the dataset, it returned other agriculturally relevant diseases. This reflects strong semantic association but weak fact-level precision.

---

### Key Takeaways

- Fine-tuning improved answer style and task alignment but did not eliminate factual errors.
- Models occasionally generated plausible but incorrect agricultural facts.
- Domain relevance does not guarantee factual accuracy.
- Fine-tuned models performed best when the required information closely matched patterns seen during training.
- These failure cases reinforce the motivation for the next phase of the project: combining fine-tuning with Retrieval-Augmented Generation (RAG) to ground responses in retrieved agricultural knowledge rather than relying solely on model memory.


## Conclusion

The original evaluation suggested that the Base model, LoRA model, and QLoRA model behaved almost identically. After rebuilding the evaluation pipeline and fixing adapter-loading contamination, the models produced clearly distinct outputs and measurable performance differences.

Key findings:

- LoRA achieved the strongest overall performance on this agricultural Q&A dataset.
- LoRA improved ROUGE-L from 0.1074 (Base) to 0.3559 and Token F1 from 0.1590 to 0.3973.
- QLoRA also improved over the Base model but did not outperform LoRA in this experiment.
- The Base model generated substantially longer responses (94.67 words on average) than the reference answers (23.62 words), indicating a mismatch in response style.
- LoRA learned the target answer style most effectively, producing answers almost identical in average length to the reference dataset.
- Fine-tuning improved task-specific behavior, but it does not guarantee factual accuracy or add new verified knowledge.
- The largest lesson from this project was that a flawed evaluation pipeline can completely hide successful training results.

Future work will focus on integrating retrieval-augmented generation (RAG), replacing the placeholder TF-IDF retriever with a domain-specific agricultural knowledge base and vector search system.