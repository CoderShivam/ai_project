# Stage 3 – Design Thinking & Architecture

## Technology Stack & Architecture

### Architecture Overview
The system follows a modular pipeline architecture:

---

### Architecture Layers

#### 1. Perception Layer – Computer Vision (Stage 1)
- Input: Webcam frame or uploaded image  
- Preprocessing:  
  - Convert to grayscale  
  - Resize to 48×48  
  - Normalize pixel values  
- Model: Custom CNN trained on FER‑2013  
- Output: Emotion label

---

#### 2. Cognitive Layer – NLP + Embeddings + RAG (Stage 2)
- Synthetic emotion‑based reviews generated manually for each emotion.  
- Embeddings created using: `sentence-transformers/all-MiniLM-L6-v2`  
- Vector search using **FAISS**  
- Sentiment scoring using **NLTK VADER**  
- Output: Retrieved relevant review with sentiment

---

#### 3. Application Layer – Real‑Time Pipeline
- Shows webcam feed  
- Detects face  
- Predicts emotion  
- Retrieves context‑aware message  

---

## Why These Technologies?

| Technology | Reason |
|-----------|--------|
| TensorFlow/Keras | Best for training CNN on FER-2013 |
| OpenCV | Real-time face detection & preprocessing |
| SentenceTransformers | High‑quality embeddings |
| FAISS | Fast vector search |
| NLTK (VADER) | Lightweight sentiment analysis |
| Python + Colab | Easy training & experimentation |

---

## 2. Data Flow & Protocol Design

### End‑to‑End Pipeline

#### 1. Image → Emotion Detection
- Webcam image captured  
- Haarcascade detects face  
- CNN predicts emotion  

---

#### 2. Emotion → Review Retrieval
- Emotion used as query key  
- Related synthetic review selected  

---

#### 3. Review → Embedding
- MiniLM generates 384‑dim embeddings  
- Stored in FAISS index  

---

#### 4. Query → Vector Search
- Retrieve top‑k nearest reviews  

---

#### 5. Response → Sentiment Analysis
- VADER validates sentiment  
- Output shown next to webcam feed  

---

## Communication
- Entire pipeline runs locally  
- No external APIs  
- Pure Python function calls  

---

## 3. RAG System Scalability

If scaling to millions of reviews:

### Recommended Upgrades
- Replace FAISS with Milvus / Pinecone  
- Batch embedding generation  
- Caching frequent queries  
- Use stronger embeddings like E5‑Large  

---

## 4. Ethics & Bias Considerations

### Risks
- Facial model trained on FER‑2013 may misclassify underrepresented faces  
- Synthetic reviews may express unintended sentiment  

---

### Mitigation
- Train on more diverse datasets (FairFace, RAF‑DB)  
- Apply sentiment filtering (VADER)  
- Add AI‑generated prediction disclaimer  

---

 
