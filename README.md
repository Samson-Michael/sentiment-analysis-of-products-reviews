# Sentiment Analysis of Products Reviews

This project focuses on classifying the sentiment of Amazon customer reviews using the VADER sentiment analysis tool from the Natural Language Toolkit and the Hugging Face RoBERTa Transformers package. Additionally, it includes a comparison of the sentiment analysis results from both tools.

# Dataset Source

We utilized a dataset of Amazon customer reviews spanning various product categories, including books, electronics, and kitchen appliances. The dataset comprises approximately 10261 reviews with ratings ranging from 1 to 5 stars.

# Preprocessing and Cleaning

The preprocessing involved handling missing values by imputing them as "missing," focusing primarily on review text. The review text and summary columns were merged into a single input to ensure consistency. A new 'sentiment' column was created based on the overall score, categorizing reviews as positive, negative, or neutral. Additionally, the time column was split into separate components for date, month, and year.


# N-gram Analysis and Handling Imbalance

In this phase of analysis, we used n-grams to analyze the text based on sentiment. The dataset has 10,261 samples and 5,000 features, corresponding to the 5,000 words considered for analysis. The target variable, 'sentiment,' was encoded from the processed reviews. To address the imbalance in the target feature, where positive sentiments dominated, SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the classes. SMOTE works by synthesizing new instances for the minority class through linear interpolation between existing examples and their k-nearest neighbors. After applying SMOTE, the dataset was balanced, and a 75:25 train-test split was used for model training and evaluation.


# Model Building: Sentiment Analysis

In the model building phase, sentiment analysis was performed, and classification metrics were evaluated. The confusion matrix was plotted to assess the model's performance, highlighting the correctly predicted records on the diagonal (2326 for negative, 2195 for neutral, and 1854 for positive). The classification report showed good precision, recall, and F1-score for all sentiment classes, with an overall accuracy of 94%. The F1-score was strong across all classes, indicating reliable classification for positive, negative, and neutral reviews. An ROC-AUC curve was plotted to visualize how well the model classified different classes, with micro and macro averages also shown to summarize the overall performance.

![image](https://github.com/user-attachments/assets/94437e1a-6efa-4956-bc2a-df2d47407d91)

![image](https://github.com/user-attachments/assets/c58f4bf6-374e-4b42-9da1-ba5299a6801b)


