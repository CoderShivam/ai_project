#!/usr/bin/env python3


import argparse
import os
import time
import json
import numpy as np
import cv2
import pandas as pd

# TF / keras
from tensorflow.keras.models import load_model

# embeddings and faiss
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize

# sentiment
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# ---- Helper functions ----
def load_keras_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at: {path}")
    print(f"[+] Loading model from: {path}")
    model = load_model(path)
    print("[+] Model loaded.")
    return model

def load_or_build_faiss(index_path, meta_path, predictions_csv=None, embed_model=None, save_if_built=True):
    """
    Loads an existing FAISS index + metadata; or builds it from predictions.csv and templated reviews.
    Returns (index, metadata_list, embed_model)
    metadata_list is a list of dicts with keys at least 'image','emotion','generated_review'
    """
    # Try to load index & metadata
    if index_path and os.path.exists(index_path) and meta_path and os.path.exists(meta_path):
        print(f"[+] Loading FAISS index from: {index_path}")
        idx = faiss.read_index(index_path)
        meta_df = pd.read_csv(meta_path)
        metadata = meta_df.to_dict(orient='records')
        print(f"[+] Loaded index with {idx.ntotal} vectors and {len(metadata)} metadata rows.")
        # ensure embed_model loaded
        if embed_model is None:
            embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        return idx, metadata, embed_model

    # Otherwise, build from predictions.csv using templates
    if predictions_csv is None or not os.path.exists(predictions_csv):
        raise FileNotFoundError("Neither existing index+metadata found nor valid predictions.csv provided to build from.")
    print(f"[+] Building FAISS index from predictions CSV: {predictions_csv}")
    df = pd.read_csv(predictions_csv)
    # Minimal templates - you can modify/expand
    templates = {
        "happy": ["I had a wonderful time — everything was great and cheerful.",
                  "This made my day; I felt very happy and pleased."],
        "sad": ["I felt disappointed and unhappy with the experience.",
                "This left me feeling down; it didn't meet my expectations."],
        "angry": ["I was upset by this and found it frustrating.",
                  "Very annoyed — this was unacceptable and angry-making."],
        "surprise": ["I was pleasantly surprised and didn't expect such results!",
                     "What a surprise — unexpectedly good and engaging."],
        "fear": ["I felt uneasy and worried about how this behaved.",
                 "This caused me concern and some anxiety."],
        "neutral": ["It was okay — nothing special, but not bad either.",
                    "A neutral experience overall; neither good nor bad."],
        "disgust": ["I found this unpleasant and off-putting.",
                    "This felt distasteful and I did not like it."]
    }

    # build reviews
    import random
    random.seed(42)
    rows = []
    for _, r in df.iterrows():
        emotion = str(r.get('prediction', 'neutral')).lower()
        if emotion not in templates:
            emotion = 'neutral'
        text = random.choice(templates[emotion])
        rows.append({
            'image': r.get('image', ''),
            'emotion': emotion,
            'generated_review': text
        })
    meta_df = pd.DataFrame(rows)

    # embeddings
    if embed_model is None:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = meta_df['generated_review'].tolist()
    print(f"[+] Computing embeddings for {len(texts)} generated reviews...")
    emb = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    emb = normalize(emb, axis=1)  # unit norm for cosine similarity via inner product

    # build faiss index
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product
    index.add(emb)
    print(f"[+] Built FAISS index with {index.ntotal} vectors.")

    if save_if_built:
        if index_path:
            faiss.write_index(index, index_path)
            print(f"[+] Saved FAISS index to {index_path}")
        if meta_path:
            meta_df.to_csv(meta_path, index=False)
            print(f"[+] Saved metadata to {meta_path}")

    return index, meta_df.to_dict(orient='records'), embed_model

def retrieve_top_k(index, metadata, embed_model, query, k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = normalize(q_emb, axis=1)
    D, I = index.search(q_emb, k)
    results = []
    for idx, score in zip(I[0], D[0]):
        m = metadata[idx]
        results.append({
            'index': int(idx),
            'score': float(score),
            'image': m.get('image'),
            'emotion': m.get('emotion'),
            'review': m.get('generated_review')
        })
    return results

def preprocess_face_for_model(roi_gray, picture_size=48):
    # expects roi_gray as grayscale image
    # resize to (48,48), normalize as in training
    roi = cv2.resize(roi_gray, (picture_size, picture_size))
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)        # batch
    roi = np.expand_dims(roi, axis=-1)       # channel
    return roi

# ---- main real-time function ----
def run_realtime(model_path,
                 cascade_path,
                 index_path=None,
                 meta_path=None,
                 predictions_csv=None,
                 labels=None,
                 top_k=3,
                 picture_size=48):
    # Load model
    model = load_keras_model(model_path)

    # Load or build FAISS index
    index, metadata, embed_model = load_or_build_faiss(index_path, meta_path, predictions_csv, embed_model=None)

    # load sentiment analyzer (VADER)
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

    # Load cascade
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade not found at {cascade_path}. Download and provide correct path.")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # default labels if not provided - must match training class order
    if labels is None:
        labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        print("[!] Using default labels order. Make sure this matches your model/class indices.")

    # start video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Make sure a camera is connected and accessible.")

    print("[+] Starting webcam. Press 'q' to quit.")
    prev_emotion = None
    last_retrieval = []
    retrieval_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Frame not grabbed, stopping.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                # enlarge box slightly for better crop
                pad = int(0.15 * w)
                x1 = max(0, x - pad); y1 = max(0, y - pad)
                x2 = min(frame.shape[1], x + w + pad); y2 = min(frame.shape[0], y + h + pad)
                roi_gray = gray[y1:y2, x1:x2]

                # preprocess and predict
                try:
                    inp = preprocess_face_for_model(roi_gray, picture_size=picture_size)
                    preds = model.predict(inp, verbose=0)[0]
                    pred_idx = int(np.argmax(preds))
                    emotion = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
                    confidence = float(preds[pred_idx])
                except Exception as e:
                    emotion = "error"
                    confidence = 0.0
                    print("Prediction error:", e)

                # If emotion changed or more than N seconds since last retrieval, query vector DB
                now = time.time()
                if emotion != prev_emotion or (now - retrieval_time) > 2.5:
                    # simple query: use emotion as query text ("I am happy") - can be improved
                    query_text = f"I am {emotion}"
                    last_retrieval = retrieve_top_k(index, metadata, embed_model, query_text, k=top_k)
                    # add sentiment for each retrieved review
                    for r in last_retrieval:
                        r['sentiment'] = sia.polarity_scores(r['review'])
                    retrieval_time = now
                    prev_emotion = emotion

                # draw rectangle and overlay text
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                label_text = f"{emotion} ({confidence:.2f})"
                cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

                # overlay top retrieved review(s) near face (or at top-left)
                # we will show only top-1 on-screen for readability
                if last_retrieval:
                    top = last_retrieval[0]
                    review_text = top['review']
                    score_text = f"score:{top['score']:.2f}"
                    senti = top.get('sentiment', {})
                    senti_str = f"sent:{senti.get('compound', 0):+.2f}"
                    overlay = f"{review_text} | {score_text} {senti_str}"
                    # draw background rectangle for readability
                    x_text = x1
                    y_text = y2 + 20
                    for i, line in enumerate([overlay[i:i+80] for i in range(0, len(overlay), 80)]):  # wrap
                        yline = y_text + i*22
                        cv2.rectangle(frame, (x_text-2, yline-18), (x_text+650, yline+6), (0,0,0), -1)
                        cv2.putText(frame, line, (x_text, yline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            cv2.imshow("Real-Time Emotion + Retrieval", frame)

            # quit on q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Exited. Cleanup done.")

# ---- CLI ----
def parse_args():
    p = argparse.ArgumentParser(description="Realtime Emotion + RAG demo")
    p.add_argument("--model_path", type=str, default="best_model.keras", help="Path to saved Keras model (.h5 or .keras)")
    p.add_argument("--cascade", type=str, default="haarcascade_frontalface_default.xml", help="Path to haarcascade XML")
    p.add_argument("--index_path", type=str, default="reviews_faiss.index", help="Path to FAISS index file (optional)")
    p.add_argument("--meta_path", type=str, default="reviews_metadata.csv", help="Path to metadata CSV for index (optional)")
    p.add_argument("--predictions_csv", type=str, default="predictions.csv", help="If index not present, build from this predictions CSV")
    p.add_argument("--top_k", type=int, default=3, help="Top-k retrieval count")
    p.add_argument("--picture_size", type=int, default=48, help="Input picture size expected by model (48 typically)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_realtime(
        model_path=args.model_path,
        cascade_path=args.cascade,
        index_path=args.index_path if args.index_path else None,
        meta_path=args.meta_path if args.meta_path else None,
        predictions_csv=args.predictions_csv,
        top_k=args.top_k,
        picture_size=args.picture_size
    )
