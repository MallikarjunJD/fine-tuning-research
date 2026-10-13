# Baseline Results — Qwen2.5-1.5B-Instruct (no fine-tuning)

Status: **COMPLETED**

This file records how the untouched base model performs on the held-out
agriculture Q&A test split, using only the same prompt template used later
for the fine-tuned model (system instruction + question). This is the
reference point every other result in this project is measured against.

## Quantitative

| Metric | Value |
|---|---|
| ROUGE-L | 0.1074 |
| BERTScore F1 | 0.8482 |
| token_f1 | 0.1590 |
| ref_contained | 0.035 |
| Exact match rate | 0.0000 |
| avg_words_pred | 94.670 |
| avg_words_ref| 23.62 |


## Qualitative examples

| # | Question | Base model answer | Reference answer |
|---|---|---|---|
| 1 | what do i look when inspect the packaging. | When inspecting packaging for your crops or products, you should focus on several key aspects to ensure quality and safety: 1. **Labeling**: Check that all labels are clear, complete, and accurate. Labels should include information such as product name, batch number, expiration date, storage instructions, and any relevant certifications. 2. **Packaging Material**: Ensure that the materials used (such as [...] | Look for signs of tampering, such as broken seals or mismatched packaging, which may indicate that the product is fake |
| 2 | which pest feed on the sap of the bean plant, causing stunting and distortion of the leaves | The pests that feed on the sap of the bean plant and cause stunting and leaf distortion are typically aphids. Aphids are small, soft-bodied insects that suck the sap from plants, which can weaken them over time. This leads to stunted growth and distorted leaves as they struggle to produce new foliage. Other common sap-sucking pests include whiteflies and mealybugs. Managing these pests often involves using [...] | Bean aphids (Aphis fabae) |
| 3 | What causes shriveling and rotting of beans seed.?. | Shriveling and rotting of bean seeds can be caused by several factors: 1. **Moisture:** Excessive moisture is a major cause. Seeds need just enough water to germinate but too much can lead to mold growth. 2. **Temperature:** Extreme temperatures, both hot and cold, can damage or kill seeds. Optimal temperature for most seeds is between 65°F (18°C) and 70°F (21°C). 3. **Disease:** Some fungi and bacteria can [...] | Fungal and Bacterial Diseases: Several fungal and bacterial pathogens can infect bean seeds and cause rotting. Common diseases include seed rot, damping-off, and seedling blight. These diseases often thrive in moist conditions and can lead to seed decay and deterioration. |
| 4 | how do you define Saturation in agriculture. | In agriculture, saturation refers to the point at which a crop or soil system is fully utilized by available water resources. At this level, there is no excess water that can be used for plant growth without causing harm. Understanding saturation helps farmers manage irrigation effectively, preventing both overwatering (which wastes water) and under-watering (which reduces crop yield). | This refers to the point at which the soil is completely filled with water, and all air spaces have been displaced. This can occur after a heavy rain or during periods of flooding. | 
| 5 | cam maize grow in clay soil | Yes, corn (maize) can grow well in clay soil if it is managed properly. Clay soils have a high water-holding capacity but may also retain too much moisture, which can lead to poor drainage and root rot issues. To optimize growth: 1. **Soil Preparation**: Improve clay soil by adding organic matter like compost or manure before planting. 2. **Water Management**: Ensure good drainage by incorporating sand or [...] | es, maize can grow in clay soil. However, clay soil has certain characteristics that can affect maize growth and productivity. Clay soil tends to retain water and drain poorly, which can lead to waterlogged conditions and root suffocation if not properly managed. Additionally, clay soil can be compacted, making it harder for roots to penetrate and access nutrients.
| 6 | Planting green manure cover crops is a great way to enhance soil fertility for several reasons: 1. **Nitrogen Fixation**: Many leguminous plants like clover, alfalfa, and vetch have the ability to fix atmospheric nitrogen from the air into the soil. This process makes available more nutrients for plant growth. 2. **Soil Structure Improvement**: Cover crops help improve soil structure by adding organic matter [...] | Planting leguminous crops such as clover, alfalfa or beans, can help fix nitrogen into the soil and add organic matter, thus improving soil fertility. |
| 7 | Which type of maize is good for storage | Maize (corn) varieties that are well-suited for long-term storage include: 1. **Dent Corn**: This variety has a hard outer layer called the dent, which helps protect the kernel from moisture loss during storage. 2. **Sweet Corn**: Sweet corn kernels have a high sugar content and can be stored longer than other types due to their natural sweetness. 3. **Flint Corn**: Flint corn kernels are harder and more [...] | The type of maize that is good for storage is one that has a high dry matter content and a hard outer layer. This helps to prevent moisture from getting into the kernel and reduces the risk of spoilage during storage | 
| 8 | which variety of maize has a hard outer layer and is typically used for animal feed. | The variety of maize known as "corn" or "maize" itself does not have a hard outer layer like some other types of corn (such as popcorn). However, if you're referring to varieties specifically bred for their tough husks that can be used for animal feed, one example would be the "Cornus" series from Pioneer Hi-Bred International. These varieties often have thick, durable hulls that make them suitable for feeding [...] | Flint maize | 

## Notes

Run `python src/evaluate.py --n_examples 500 --output_json results/baseline_eval.json`
(no `--adapter_path`) to populate this file with measured numbers.
