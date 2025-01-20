# Finetune Hugging Face Model & Host As A Realtime Endpoint on Sagemaker

In 2021, during a project at Explore, our team worked on classifying tweets into one of four categories related to man-made climate change: **"Negative"**, **"Pro"**, **"Neutral"**, or **"News"**. At the time, we relied on traditional NLP techniques such as lowercasing, removing stopwords, and lemmatization with part-of-speech tagging. To convert text into usable vectors, we used methods like **Bag of Words** and **Word2Vec**. For classification, we applied algorithms such as **k-Nearest Neighbors**, **Random Forest**, **XGBoost**, and **Logistic Regression**. Our efforts resulted in a **macro-weighted F1 score** of approximately **0.70** and an **accuracy** of around **0.77**.

Fast forward four years, I became curious about how transformer-based techniques would perform compared to our earlier approach. To explore this, I revisited the original dataset and used a **RoBERTa model** as a baseline, fine-tuning it to classify the tweets effectively. Fine-tuning on RoBERTa yielded similar results with **minimal data preprocessing**.

This project demonstrates an **end-to-end machine learning workflow**, incorporating **SageMaker processing, SageMaker training jobs, and SageMaker endpoints**. It concludes with deploying the fine-tuned model as a **real-time endpoint** for practical use.


## Key takeaways from the project:

### **Reduced Preprocessing with RoBERTa**
One of the major advantages of using **RoBERTa** was its ability to achieve similar performance to the previously used approach with **much less data preprocessing**.  
Traditional models required extensive text cleaning steps such as **tokenization, lemmatization, and stopword removal**. In contrast, RoBERTa required **minimal intervention**, allowing more focus on fine-tuning the model and improving overall efficiency.

### **Efficiency Gains with GPU Acceleration**
The use of **GPU instances** significantly reduced training time. For example, when training with a **CPU instance**, it took **2 hours** to process **4,000 samples**. However, when switched to a **GPU instance**, training on **16,000 samples** was completed in a fraction of the time (**20 minutes**) — demonstrating the immense value of **GPU acceleration** for deep learning tasks.  
This highlights how hardware choices can have a profound impact on the **scalability and efficiency** of model training.

### **Streamlined Workflow with Hugging Face Deep Learning Containers**
Another key takeaway from the project was the ease of use provided by **Hugging Face Deep Learning Containers (DLCs)**. These containers come with **pre-installed packages and libraries** commonly used in NLP tasks, such as **Transformers** and **PyTorch**, eliminating the need for complex setup and configuration.  
This allowed me to focus more on **model development** and less on managing dependencies, making the overall preprocessing and training process **much smoother**.
