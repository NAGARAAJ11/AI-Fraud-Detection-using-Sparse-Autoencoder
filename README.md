\# AI Fraud Detection using Sparse Autoencoder



\## 📌 Project Overview



Financial fraud detection is a critical challenge in modern banking and online transactions. Traditional rule-based systems often fail to detect new or rare fraud patterns.



This project implements an \*\*unsupervised deep learning approach\*\* using a \*\*Sparse Autoencoder\*\* to detect fraudulent credit card transactions based on reconstruction error. The model learns normal transaction patterns and flags anomalies as potential fraud.



The system includes a \*\*Streamlit web application\*\* that allows users to upload transaction data and view fraud detection results in real time with performance metrics and visualizations.



\---



\## 🎯 Problem Statement



Detect fraudulent credit card transactions using machine learning techniques capable of identifying anomalies in highly imbalanced datasets.



Fraudulent transactions are rare compared to legitimate ones, making supervised learning challenging. Therefore, this project uses an \*\*unsupervised anomaly detection method\*\* to identify suspicious activity without requiring labeled fraud examples during training.



\---



\## 🚀 Features



\* Sparse Autoencoder for anomaly detection

\* Real-time fraud prediction

\* Interactive Streamlit web interface

\* ROC Curve and Precision-Recall visualization

\* Model performance evaluation (Accuracy, F1 Score, AUC)

\* Scalable and production-ready structure

\* Dataset handling with large file exclusion using `.gitignore`



\---



\## 🧠 Model Used



\### Sparse Autoencoder



A Sparse Autoencoder is a neural network trained to reconstruct input data while enforcing sparsity constraints on hidden layer activations. This helps the model learn meaningful patterns and detect anomalies effectively.



\### Workflow



1\. Load dataset

2\. Preprocess data

3\. Train Sparse Autoencoder

4\. Compute reconstruction error

5\. Set anomaly threshold

6\. Detect fraudulent transactions

7\. Visualize performance metrics



\---



\## 🛠️ Tech Stack



\*\*Programming Language\*\*



\* Python



\*\*Libraries\*\*



\* Streamlit

\* NumPy

\* Pandas

\* Scikit-learn

\* TensorFlow

\* Matplotlib

\* Joblib



\*\*Machine Learning\*\*



\* Deep Learning

\* Anomaly Detection

\* Sparse Autoencoder



\---



\## 📂 Project Structure



```

AI-Fraud-Detection-using-Sparse-Autoencoder/



app.py

train\_model.py

requirements.txt

README.md

.gitignore



dataset/

&#x20;  creditcard.csv   (ignored due to large size)



models/

&#x20;  model.h5

&#x20;  scaler.pkl



utils/

&#x20;  data\_preprocessing.py

&#x20;  evaluation\_metrics.py

```



\---



\## 📊 Dataset



\*\*Dataset Name:\*\* Credit Card Fraud Detection Dataset



\*\*Description:\*\*

The dataset contains anonymized credit card transactions made by European cardholders. It includes legitimate and fraudulent transactions with highly imbalanced classes.



\*\*Important Note:\*\*

The dataset file (`creditcard.csv`) is larger than \*\*100 MB\*\*, so it is excluded from the repository using `.gitignore`.



\---



\## ⚙️ Installation



\### Step 1 — Clone the Repository



```

git clone https://github.com/NAGARAAJ11/AI-Fraud-Detection-using-Sparse-Autoencoder.git

```



\### Step 2 — Navigate to Project Directory



```

cd AI-Fraud-Detection-using-Sparse-Autoencoder

```



\### Step 3 — Install Dependencies



```

pip install -r requirements.txt

```



\---



\## ▶️ Running the Application



Run the Streamlit application:



```

streamlit run app.py

```



The application will open in your browser.



\---



\## 📈 Model Evaluation Metrics



The model performance is evaluated using the following metrics:



\* Accuracy

\* Precision

\* Recall

\* F1 Score

\* ROC Curve

\* Precision-Recall Curve

\* AUC Score



These metrics help measure the effectiveness of the fraud detection system in identifying fraudulent transactions while minimizing false alarms.



\---



\## 📊 Sample Output



The system provides:



\* Fraud detection predictions

\* Confusion matrix

\* ROC curve visualization

\* Precision-Recall curve

\* Transaction classification results



\---



\## 🖥️ Screenshots



Add screenshots of your Streamlit application here.



Example:



```

assets/

&#x20;  dashboard.png

&#x20;  roc\_curve.png

&#x20;  prediction\_result.png

```



\---



\## 🔮 Future Improvements



\* Deploy the application to cloud platforms

\* Add real-time transaction monitoring

\* Implement additional anomaly detection models:



&#x20; \* Isolation Forest

&#x20; \* One-Class SVM

\* Improve model accuracy using hyperparameter tuning

\* Add API integration for production use



\---



\## 📌 Use Cases



\* Banking fraud detection

\* Online payment security

\* Financial risk monitoring

\* Anomaly detection in transactions

\* Cybersecurity systems



\---



\## 👨‍💻 Author



\*\*Nagaraj D\*\*

B.Tech — Artificial Intelligence and Data Science

Coimbatore Institute of Technology



\---



\## 📜 License



This project is for educational and research purposes.



\---



\## ⭐ If you found this project useful



Please consider giving it a star on GitHub.



