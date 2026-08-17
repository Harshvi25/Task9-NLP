# Spam SMS Classification using NLP

## Overview

This project classifies SMS messages as **Ham (normal)** or **Spam** using Natural Language Processing (NLP).

## Dataset

- Dataset: `spam.csv`
- Total messages: 5572
- After removing duplicates: 5169
- Ham: 4516
- Spam: 653

## Steps

1. Load the dataset
2. Remove unnecessary columns
3. Rename columns
4. Check missing values and duplicates
5. Remove duplicate messages
6. Clean the text:
   - Convert to lowercase
   - Remove numbers
   - Remove punctuation
   - Remove extra spaces
7. Encode labels:
   - Ham = 0
   - Spam = 1
8. Split data into training and testing sets
9. Convert text into numerical features using TF-IDF
10. Train a Logistic Regression model
11. Evaluate the model
12. Save the trained model and TF-IDF vectorizer

## Model

**Logistic Regression**

## Author
Harshvi Patel

```text
[[902   1]
 [ 43  88]]
