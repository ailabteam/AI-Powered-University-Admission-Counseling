# src/build_knowledge_base.py

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
import time

def main():
    """
    Main function to build the knowledge base from raw data.
    - Processes the Excel file.
    - Creates embeddings using a sentence transformer model.
    - Builds and saves a FAISS index and the corresponding context data.
    """
    
    # --- Configuration ---
    # Build absolute paths from the script's location
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SRC_DIR)

    RAW_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/raw/dut_faq.xlsx')
    PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, 'data/processed/dut_faq_processed.csv')
    
    MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models')
    FAISS_INDEX_PATH = os.path.join(MODEL_OUTPUT_DIR, 'faq_index.faiss')
    CONTEXTS_PATH = os.path.join(MODEL_OUTPUT_DIR, 'faq_contexts.json')

    EMBEDDING_MODEL_NAME = 'bkai-foundation-models/vietnamese-bi-encoder'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("--- Knowledge Base Builder ---")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Using device: {DEVICE}")
    
    # Create necessary directories
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # --- Step 1: Process Data ---
    print(f"\n[Step 1/4] Processing data from: {RAW_DATA_PATH}")
    start_time = time.time()
    try:
        df = pd.read_excel(RAW_DATA_PATH)
        df['question'] = df['question'].astype(str).fillna('')
        df['answers'] = df['answers'].astype(str).fillna('')
        df['context'] = "Hỏi: " + df['question'] + "\nĐáp: " + df['answers']
        df.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"  -> Data processed successfully. Shape: {df.shape}")
        print(f"  -> Saved processed data to: {PROCESSED_DATA_PATH}")
    except FileNotFoundError:
        print(f"  -> ERROR: Raw data file not found at {RAW_DATA_PATH}. Please check the path.")
        return
    except Exception as e:
        print(f"  -> ERROR during data processing: {e}")
        return
    print(f"  -> Time taken: {time.time() - start_time:.2f} seconds")

    # --- Step 2: Load Embedding Model ---
    print(f"\n[Step 2/4] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    start_time = time.time()
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)
        print("  -> Embedding model loaded successfully.")
    except Exception as e:
        print(f"  -> ERROR loading model: {e}")
        return
    print(f"  -> Time taken: {time.time() - start_time:.2f} seconds")

    # --- Step 3: Create Embeddings ---
    contexts = df['context'].tolist()
    print(f"\n[Step 3/4] Creating embeddings for {len(contexts)} contexts...")
    start_time = time.time()
    try:
        embeddings = model.encode(contexts, show_progress_bar=True, convert_to_tensor=True)
        embeddings_np = embeddings.cpu().numpy()
        print(f"  -> Embeddings created successfully. Shape: {embeddings_np.shape}")
    except Exception as e:
        print(f"  -> ERROR creating embeddings: {e}")
        return
    print(f"  -> Time taken: {time.time() - start_time:.2f} seconds")

    # --- Step 4: Build and Save FAISS Index ---
    d = embeddings_np.shape[1]
    print(f"\n[Step 4/4] Building and saving FAISS index (dimension: {d})...")
    start_time = time.time()
    try:
        index = faiss.IndexFlatL2(d)
        if DEVICE == 'cuda':
            print("  -> Using GPU for FAISS index construction.")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(embeddings_np)
        print(f"  -> Total vectors in index: {index.ntotal}")

        if DEVICE == 'cuda':
            index_cpu = faiss.index_gpu_to_cpu(index)
            faiss.write_index(index_cpu, FAISS_INDEX_PATH)
        else:
            faiss.write_index(index, FAISS_INDEX_PATH)
            
        print(f"  -> FAISS index saved to: {FAISS_INDEX_PATH}")

        with open(CONTEXTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(contexts, f, ensure_ascii=False, indent=2)
        print(f"  -> Contexts saved to: {CONTEXTS_PATH}")
    except Exception as e:
        print(f"  -> ERROR building/saving FAISS index: {e}")
        return
    print(f"  -> Time taken: {time.time() - start_time:.2f} seconds")

    print("\n--- Knowledge Base built successfully! ---")

if __name__ == "__main__":
    main()
