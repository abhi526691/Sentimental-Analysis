# Sentiment Analysis on Twitter Data

## Overview
This project focuses on sentiment analysis using a dataset of approximately half a million tweets. The goal is to classify the sentiment of tweets into five distinct categories:

- **Strong_Pos**
- **Neutral**
- **Mild_Pos**
- **Strong_Neg**
- **Mild_Neg**

Various traditional machine learning models and deep learning architectures were employed to achieve this task. Given the dataset size, **LinearSVC** was selected due to its computational efficiency, though **SVM** is recommended for smaller datasets due to its stronger performance in non-linear tasks.

The models are evaluated using vectorization techniques like **TF-IDF**, **CountVectorizer**, **Word2Vec**, **GloVe**, and **FastText**, while deep learning models like **RNN**, **LSTM**, and **GRU** are also employed. 

Contributions and feedback from the community are welcome!

---

## Table of Contents
- [Models Used](#models-used)
- [Dataset](#dataset)
- [Preprocessing Steps](#preprocessing-steps)
- [Future Work](#future-work)
- [Contributing](#contributing)

---

## Models Used

Below is a summary of the models and techniques employed. Contributors can add accuracy and evaluation metrics after running their own experiments.

| **Model**                | **Vectorization Technique**  | **Classifier**      | **Accuracy** |
|--------------------------|------------------------------|---------------------|--------------|
| TF-IDF(UNIGRAM)                | TF-IDF Vectorizer            | LinearSVC           |   64 with overfitting          |
| TF-IDF(BIGRAM)                   | TF-IDF Vectorizer            | LinearSVC           |   60 with overfitting           |
| CountVectorizer           | CountVectorizer              | LinearSVC           |  73 with overfitting          |
| Word2Vec (CBOW) | Word2Vec                     | LinearSVC           |        61 with overfitting     |
| Word2Vec (Skipgram) | Word2Vec                | LinearSVC           |       61 with overfitting       |
| GloVe                    | GloVe Embeddings             | LinearSVC           |     58 No overfitting         |
| FastText                 | FastText Embeddings          | LinearSVC           |    61 No overfitting          |
| RNN                      | Tokenizer                          | RNN                 |     66 on 1/10th data         |
| LSTM                     | Tokenizer                            | LSTM                |     68 on 1/15th data         |


---

## Dataset

- **Size**: ~500,000 tweets
- **Classes**: 
  - Strong_Pos
  - Neutral
  - Mild_Pos
  - Strong_Neg
  - Mild_Neg

---

## Preprocessing Steps

The following preprocessing techniques were used to prepare the text data for analysis:

1. **Handle Emojis, Slangs, Punctuation, and Short Forms**  
   Normalize emojis and convert slangs or short forms to their formal equivalents.
2. **Spelling Corrections**  
   Correct misspelled words to ensure consistency.
3. **POS Tagging**  
   Use part-of-speech tagging to assist with context-specific word corrections.
4. **Handling Pronouns and Special Characters**  
   Normalize or remove special characters and handle pronouns effectively.
5. **Tokenization**  
   Tokenize the text into words or symbols.
6. **Remove Stop Words**  
   Eliminate common stop words that don't contribute to sentiment analysis.
7. **Negation Handling**  
   Properly account for negation in phrases like "not happy" to ensure accurate sentiment detection.
8. **Stemming and Lemmatization**  
   Reduce words to their base form using stemming or lemmatization.
9. **Lowercasing and En-grams**  
   Convert text to lowercase and use both unigrams and bigrams to capture relationships between words.

---

## Future Work

- Experiment with other vectorization techniques or deep learning models.
- Explore different hyperparameters or ensemble approaches to improve model accuracy.
- Improve the preprocessing pipeline by adding additional steps or features.
- Consider using sentiment lexicons or contextual embeddings (e.g., BERT).
- Add results and analyses for other classification metrics beyond accuracy, such as precision, recall, and F1-score.

---

## Contributing

We welcome contributions from the community! Here's how you can help:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and add relevant documentation.
4. Push your branch and open a pull request.

Feel free to comment on issues or suggest new features. Your feedback and contributions are highly appreciated!

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.