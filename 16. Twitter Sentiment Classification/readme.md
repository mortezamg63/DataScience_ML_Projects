
his code implements a Twitter sentiment analysis project using a combination of traditional machine learning models and a custom-built neural network. The goal is to classify tweets into one of four sentiment categories: Positive, Negative, Neutral, or Irrelevant based on the content of the tweet.

The project begins with data loading and preprocessing, where the text data is cleaned and transformed into numerical format using TF-IDF vectorization. Exploratory data analysis (EDA) is then performed to visualize sentiment distribution, entity frequencies, and message lengths. This helps in understanding the dataset's structure and preparing for model building.

Several classification models are implemented and evaluated:

    k-Means Clustering is used unsupervised to identify natural clusters in the data.

    Logistic Regression achieves around 82% accuracy, offering a simple and interpretable baseline.

    Random Forest Classifier performs best among classical models with 94% accuracy, indicating strong predictive power.

    Decision Tree Classifier provides 89% accuracy, balancing performance and interpretability.

A custom Artificial Neural Network (ANN) is then built using PyTorch, featuring two hidden layers with dropout for regularization. The network is trained over 10 epochs and achieves an impressive 97% accuracy on the validation set, outperforming the classical models. It also provides class-wise accuracy metrics to assess individual class performance.

Overall, this project demonstrates a full machine learning pipeline for natural language processing (NLP) tasks: from data loading and EDA to vectorization, model training, evaluation, and deep learning integration for optimal performance.

The table below shows the ML algorithms and their performance on this dataset:

| Algorithm | Acc |
|----------|------|
|Logistic Regression| 81.68%|
|Random Forest | 93.894%|
|Decision Tree | 88.69%|
|ANN | 97.4%|
