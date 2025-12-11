#  Plagiarism Detection using NLP & Machine Learning

This project is an end-to-end plagiarism detection system built using **Natural Language Processing (NLP)** and **Machine Learning**.  
It analyzes two text documents, converts them into TF-IDF vectors, and uses a trained ML classifier to determine whether plagiarism exists and how similar the documents are.

---

##  Features

- Detects plagiarism between any two input documents  
- Uses **TF-IDF Vectorization** for text representation  
- Built with a trained ML model (SVM/Logistic Regression)  
- Includes complete data preprocessing and EDA notebooks  
- Provides similarity score + plagiarism prediction  
- Model and vectorizer saved as `.pkl` files for reuse  

---

##  Technologies Used

- Python  
- Jupyter Notebook  
- NLP (Tokenization, Lemmatization, Stopword removal)  
- TF-IDF Vectorizer  
- Machine Learning (Classification)  
- Scikit-learn  
- Pandas, NumPy  

---

##  Project Structure
plagiarism_detection/
│
├── data_exploration.ipynb # Exploratory data analysis
├── data_preprocessing.ipynb # Text cleaning & preprocessing
├── data_set_training.ipynb # Model training
├── code_plagiarism.ipynb # Final plagiarism detection module
│
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── model.pkl # Trained ML model
├── IR-Plag-Dataset.zip # Dataset used for training
│
├── input.rtf # Sample input file 1
├── input1.rtf # Sample input file 2




---

## 🔧 How It Works

1. Input documents are read (RTF/Text/PDF converted to text).  
2. Text is cleaned using NLP preprocessing.  
3. TF-IDF converts documents to numerical vectors.  
4. The trained ML model predicts similarity and plagiarism.  
5. Output shows:
   - Similarity score  
   - Plagiarism probability  
   - Plagiarized / Not Plagiarized  

---

##  Example Output
Similarity Score: 0.78
Prediction: HIGH PLAGIARISM DETECTED
---

## 📥 Dataset

The model is trained on the **IR-Plag Dataset**, which contains labeled pairs of plagiarized and non-plagiarized documents.

---

## 📎 Future Enhancements

- Add semantic similarity using BERT embeddings  
- Add UI/Flask web interface  
- Improve accuracy with ensemble models  
- Support PDF extraction natively  

---

## 🧑‍💻 Author

**Vinay Kumar**  
NLP & Machine Learning Enthusiast  

