---
base_model: meta-llama/Llama-3.2-11B-Vision-Instruct
language:
- en
library_name: transformers
license: llama3.2
tags:
- llama-3
- llama
- meta
- facebook
- unsloth
- transformers
- multimodal
- vision
- pytorch
---

## ***See [our collection](https://huggingface.co/collections/unsloth/vision-multimodal-models-673eb9908fc2cb3deebd2fa3) for vision models including Llama 3.2, Llava, Qwen2-VL and Pixtral.***


# Finetune Llama 3.2, Gemma 2, Mistral 2-5x faster with 70% less memory via Unsloth!

We have a free Google Colab Tesla T4 notebook for Llama 3.2 Vision (11B) here: https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/Discord%20button.png" width="200"/>](https://discord.gg/unsloth)
[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)

# unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit
For more details on the model, please go to Meta's original [model card](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)

## ✨ Finetune for Free

All notebooks are **beginner friendly**! Add your dataset, click "Run All", and you'll get a 2x faster finetuned model which can be exported to GGUF, vLLM or uploaded to Hugging Face.

| Unsloth supports          |    Free Notebooks                                                                                           | Performance | Memory use |
|-----------------|--------------------------------------------------------------------------------------------------------------------------|-------------|----------|
| **Llama-3.2 (3B)**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)               | 2.4x faster | 58% less |
| **Llama-3.2 (11B vision)**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1j0N4XTY1zXXy7mPAhOC1_gMYZ2F2EBlk?usp=sharing)               | 2x faster | 60% less |
| **Qwen2 VL (7B)**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1whHb54GNZMrNxIsi2wm2EY_-Pvo2QyKh?usp=sharing)               | 1.8x faster | 60% less |
| **Qwen2.5 (7B)**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1Kose-ucXO1IBaZq5BvbwWieuubP7hxvQ?usp=sharing)               | 2x faster | 60% less |
| **Llama-3.1 (8B)**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing)               | 2.4x faster | 58% less |
| **Phi-3.5 (mini)** | [▶️ Start on Colab](https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing)               | 2x faster | 50% less |
| **Gemma 2 (9B)**      | [▶️ Start on Colab](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing)               | 2.4x faster | 58% less |
| **Mistral (7B)**    | [▶️ Start on Colab](https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing)               | 2.2x faster | 62% less |
| **DPO - Zephyr**     | [▶️ Start on Colab](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing)               | 1.9x faster | 19% less |

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/documentation%20green%20button.png" width="200"/>](https://docs.unsloth.ai)

- This [conversational notebook](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing) is useful for ShareGPT ChatML / Vicuna templates.
- This [text completion notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing) is for raw text. This [DPO notebook](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing) replicates Zephyr.
- \* Kaggle has 2x T4s, but we use 1. Due to overhead, 1x T4 is 5x faster.

## Special Thanks
A huge thank you to the Meta and Llama team for creating and releasing these models.

## Model Information

The Meta Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction-tuned generative models in 1B and 3B sizes (text in/text out). The Llama 3.2 instruction-tuned text only models are optimized for multilingual dialogue use cases, including agentic retrieval and summarization tasks. They outperform many of the available open source and closed chat models on common industry benchmarks.

**Model developer**: Meta

**Model Architecture:** Llama 3.2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety.

**Supported languages:**  English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai are officially supported. Llama 3.2 has been trained on a broader collection of languages than these 8 supported languages. Developers may fine-tune Llama 3.2 models for languages beyond these supported languages, provided they comply with the Llama 3.2 Community License and the Acceptable Use Policy. Developers are always expected to ensure that their deployments, including those that involve additional languages, are completed safely and responsibly.

**Llama 3.2 family of models** Token counts refer to pretraining data only. All model versions use Grouped-Query Attention (GQA) for improved inference scalability.

**Model Release Date:** Sept 25, 2024

**Status:** This is a static model trained on an offline dataset. Future versions may be released that improve model capabilities and safety.

**License:** Use of Llama 3.2 is governed by the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) (a custom, commercial license agreement).

Where to send questions or comments about the model Instructions on how to provide feedback or comments on the model can be found in the model [README](https://github.com/meta-llama/llama3). For more technical information about generation parameters and recipes for how to use Llama 3.1 in applications, please go [here](https://github.com/meta-llama/llama-recipes). 
