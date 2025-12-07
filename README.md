# 🎭 EmotionSense – Real-Time Emotion Detection & RAG System

EmotionSense is an end-to-end AI system that combines **real-time facial emotion detection** with a lightweight **Retrieval-Augmented Generation (RAG)** pipeline to generate contextual emotion-based responses.  
This project was developed as part of the **Junior AI Engineer Assignment**.

---

## ✔️ Project Overview

EmotionSense consists of **three integrated stages**:

---

### ⭐ **Stage 1 – Facial Emotion Recognition**

A CNN model trained on the **FER-2013** dataset predicts 7 emotions:

`Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`

**Features:**
- Built using **TensorFlow/Keras**
- Real-time detection using **OpenCV Haarcascade**
- Preprocessing pipeline for 48×48 grayscale face crops
- Outputs emotion label in real time during webcam inference

---

### ⭐ **Stage 2 – Embeddings, Sentiment & RAG Retrieval**

This stage adds an NLP pipeline that generates emotion-based text responses.

**Components:**

#### 🔹 **Embeddings**
- Generated using **Sentence-Transformers (all-MiniLM-L6-v2)**
- Stored using **FAISS** vector index
- Metadata stored in CSV (image name, emotion label, review text)

#### 🔹 **Sentiment Analysis**
- Uses **NLTK VADER** to score retrieved reviews

#### 🔹 **Retrieval-Augmented Generation (RAG)**
- Retrieves **top-k relevant reviews** based on predicted emotion
- Combines review + sentiment to display contextual messages

---

### ⭐ **Stage 3 – Architecture & System Design**

A detailed explanation of:
- Tech stack reasoning  
- Data flow (image → emotion → embeddings → retrieval → response)  
- Scalability considerations  
- Ethical implications & bias mitigation  

Full document here: **Stage3_Design_Architecture.md**

---

## 🧪 Real-Time Demo Workflow

The real-time script performs:

1. Opens webcam  
2. Detects face using Haarcascade  
3. Predicts emotion using trained CNN  
4. Retrieves `top-k` reviews from FAISS  
5. Applies **VADER sentiment scoring**  
6. Displays **emotion + contextual message** on the webcam window  

---

## 📁 Project Structure
ai_project/
│── best_model.keras # Trained CNN model

│── facer_ipynb_.ipynb # Notebook with Stage 1 + Stage 2 workflow

│── haarcascade_frontalface_default.xml

│── predictions.csv # Stage 1 predictions

│── reviews_faiss.index # FAISS vector index

│── reviews_metadata.csv # Embedding metadata

│── newscript.py # Real-time demo script

│── Stage3_Design_Architecture.md # Stage 3 system design

│── README.md # You are reading this file


---

## 🛠 Technologies Used

### 🔹 **Computer Vision**
- TensorFlow / Keras  
- OpenCV  
- Custom CNN model (trained on FER-2013)

### 🔹 **NLP & Retrieval**
- Sentence-Transformers (MiniLM)
- FAISS vector search
- NLTK VADER sentiment analysis

### 🔹 **Tools**
- Google Colab (training & embeddings)
- Python 3.x  
- GitHub (version control)

---

## ⚙️ Installation Guide

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
2️⃣ Install Dependencies
pip install tensorflow opencv-python sentence-transformers faiss-cpu nltk pandas

3️⃣ Download NLTK Lexicon
import nltk
nltk.download("vader_lexicon")

▶️ Usage
Run real-time webcam emotion + RAG response:
python newscript.py --model_path best_model.keras --cascade haarcascade_frontalface_default.xml --index_path reviews_faiss.index --meta_path reviews_metadata.csv:
A window will open showing:
Detected emotion
Green bounding box
Text response retrieved from RAG pipeline
