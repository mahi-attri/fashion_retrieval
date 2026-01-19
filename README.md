# Multimodal Fashion & Context Retrieval System

This project implements a zero-shot multimodal fashion retrieval system that improves upon vanilla CLIP-based image–text retrieval by explicitly modeling fashion attributes such as clothing type, color, environment, and style.

---

## Problem Overview

Standard image–text retrieval systems often fail to handle compositional fashion queries like:
- “Person wearing a blue jacket outdoors”
- “Business attire in an office”
- “Casual outfit for a city walk”

This system addresses these limitations by incorporating attribute-aware representations into both indexing and retrieval.

---

## System Architecture

1. **Image Encoding**  
   Images are encoded using the CLIP image encoder.

2. **Attribute Inference**  
   Zero-shot attribute inference is performed using a fixed vocabulary of colors, clothing items, environments, and styles.

3. **Attribute Fusion**  
   Image embeddings are fused with inferred attribute embeddings to create a compositional representation.

4. **Retrieval & Re-ranking**  
   FAISS is used for fast similarity search, followed by a re-ranking stage that combines image similarity, fused similarity, and attribute-level signals.

---

## Installation

```
pip install -r requirements.txt

python indexer.py --image_dir path/to/images --output_dir ./index_data

python retriever.py --index_dir ./index_data --query "A person wearing a blue jacket outdoors" --k 5

