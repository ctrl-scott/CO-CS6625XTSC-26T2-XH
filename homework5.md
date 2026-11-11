# Homework 5: Decision Tree Phishing Detection

**Machine Learning-Based Phishing URL Detection and Feature Analysis**

**Charles S. Owen**  
Department of Computer Science, Troy University  
CS6625XTSC-26T2 — Machine Learning-Based Phishing URL Detection and Feature Analysis  
Dr. Xiaoli Huan  
November 15, 2026

---

## Preliminary Setup and Dataset Verification

### Package installation

```bash
pip install ucimlrepo
pip install scikit-learn
```

The captured environment reported `ucimlrepo 0.0.7`, `pandas 2.2.3`, `numpy 2.4.6`, and `scikit-learn 1.6.1` as already installed.

### Imports

```python
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
```

The output notes that earlier `X.shape` and `y.shape` variables were renamed `X_raw` and `y_original`.

### Data dimensions

| Data | Shape |
|---|---|
| Original features (`X_raw`) | `(235795, 54)` |
| Original targets (`y_original`) | `(235795,)` |
| Prepared features (`X`) | `(235795, 50)` |

The prepared feature dataset and target each have **235,795 rows**, with **0 missing values** reported.

### Prepared feature columns

```text
URLLength, DomainLength, IsDomainIP, URLSimilarityIndex,
CharContinuationRate, TLDLegitimateProb, URLCharProb, TLDLength,
NoOfSubDomain, HasObfuscation, NoOfObfuscatedChar, ObfuscationRatio,
NoOfLettersInURL, LetterRatioInURL, NoOfDegitsInURL, DegitRatioInURL,
NoOfEqualsInURL, NoOfQMarkInURL, NoOfAmpersandInURL,
NoOfOtherSpecialCharsInURL, SpacialCharRatioInURL, IsHTTPS, LineOfCode,
LargestLineLength, HasTitle, DomainTitleMatchScore, URLTitleMatchScore,
HasFavicon, Robots, IsResponsive, NoOfURLRedirect, NoOfSelfRedirect,
HasDescription, NoOfPopup, NoOfiFrame, HasExternalFormSubmit,
HasSocialNet, HasSubmitButton, HasHiddenFields, HasPasswordField,
Bank, Pay, Crypto, HasCopyrightInfo, NoOfImage, NoOfCSS, NoOfJS,
NoOfSelfRef, NoOfEmptyRef, NoOfExternalRef
```

The preliminary analysis selected `URLLength`, `DomainLength`, `URLSimilarityIndex`, `CharContinuationRate`, and `TLDLegitimateProb`, and referenced `X[selected_features].describe().T` for descriptive statistics. No numerical descriptive-statistics table was included in the source output.

---

## Part 1: Prepared Data

### 1.1 Training and testing split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

The dataset was split into approximately **80% training** and **20% testing** observations.

| Variable | Shape | Observations |
|---|---|---:|
| `X_train` | `(188636, 50)` | 188,636 |
| `X_test` | `(47159, 50)` | 47,159 |
| `y_train` | `(188636,)` | 188,636 |
| `y_test` | `(47159,)` | 47,159 |

**Verification:** `188636 + 47159 = 235795`.

These train/test variables are retained for the remainder of the course experiment.

### 1.2 Recoding targets

The output maps the binary targets as follows:

| Target value | Label | Total count |
|---|---|---:|
| `0` | Legitimate | 134,850 |
| `1` | Phishing | 100,945 |
| **Total** | | **235,795** |

```python
label_mapping = {0: "Legitimate 0", 1: "Phishing 1"}
label_counts = y.value_counts().sort_index()
label_counts.index = label_counts.index.map(label_mapping)
print(label_counts)
```

```text
Legitimate 0    134850
Phishing 1      100945
Name: count, dtype: int64
```

---

## Part 2: Train a Decision Tree

### 2.1 Create, fit, and use the model

```python
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

print("printing dt_pred[:20] with labeling 0 = legitimate and 1 = phishing")
for value in dt_pred[:20]:
    if value == 0:
        print(f"{value} : Legitimate")
    else:
        print(f"{value} : Phishing")
```

### 2.2 First 20 predictions

```text
0 : Legitimate
1 : Phishing
0 : Legitimate
1 : Phishing
0 : Legitimate
0 : Legitimate
1 : Phishing
1 : Phishing
0 : Legitimate
0 : Legitimate
1 : Phishing
0 : Legitimate
1 : Phishing
0 : Legitimate
1 : Phishing
0 : Legitimate
0 : Legitimate
1 : Phishing
0 : Legitimate
1 : Phishing
```

The captured output additionally describes a **model depth of 4** and **5 leaf nodes** for the model configured with `max_depth=5`. These are retained as reported, without re-running the model.

---

## Part 3: Evaluate the Model

### 3.1 Calculate the evaluation metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dt_accuracy = accuracy_score(y_test, dt_pred)
dt_precision = precision_score(y_test, dt_pred)
dt_recall = recall_score(y_test, dt_pred)
dt_f1 = f1_score(y_test, dt_pred)

print(f"Accuracy : {dt_accuracy:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall   : {dt_recall:.4f}")
print(f"F1-score : {dt_f1:.4f}")
```

### 3.2 Reported results

| Metric | Reported value |
|---|---:|
| Accuracy | 1.0000 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1-score | 1.0000 |

All four values are **as reported in the uploaded output**; the Markdown conversion does not independently verify the experiment.

### 3.3 Interpretation of evaluation metrics

**Accuracy:** Of all test URLs, what proportion were classified correctly? Conceptually, correct predictions divided by all predictions. Accuracy alone does not reveal which mistakes were made, an important distinction in cybersecurity.

**Precision:** When the decision tree predicts phishing, how often is it actually phishing? Conceptually, correct phishing predictions divided by all predicted phishing URLs. Low precision corresponds to many false alarms.

**Recall:** Of all actual phishing URLs, how many were detected? Conceptually, detected phishing URLs divided by all actual phishing URLs. Low recall can allow phishing sites to escape detection.

**F1-score:** A single measure of the balance between precision and recall. A higher F1-score generally indicates a better balance between false alerts and missed phishing URLs.

**Accuracy versus recall example:** A model could have 99% accuracy and only 50% recall; it might therefore miss approximately half of the phishing URLs. High accuracy alone does not prove strong phishing protection.

---

## Part 4: Confusion Matrix

### 4.1 Generate the confusion matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, dt_pred)
print(cm)
```

The labels are `0 = legitimate` and `1 = phishing`.

### 4.2 Reported confusion matrix

| Actual / Predicted | Predicted Legitimate | Predicted Phishing |
|---|---:|---:|
| **Actual Legitimate** | 26,970 (TN) | 0 (FP) |
| **Actual Phishing** | 0 (FN) | 20,189 (TP) |

| Outcome | Count |
|---|---:|
| True negatives (TN) | 26,970 |
| False positives (FP) | 0 |
| False negatives (FN) | 0 |
| True positives (TP) | 20,189 |

The four confusion-matrix cells sum to **47,159**, matching the test-set size.

### 4.3 Plot the confusion matrix

```python
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Legitimate", "Phishing"]
).plot()

plt.title("Decision Tree Confusion Matrix")
plt.show()
```

*The uploaded text contains the commands to plot the figure but not an embedded image of the plotted confusion matrix.*

### 4.4 Cybersecurity interpretation

- **True positive:** A phishing URL was correctly classified as phishing.
- **True negative:** A legitimate URL was correctly classified as legitimate.
- **False positive:** A legitimate URL was incorrectly classified as phishing. Possible consequences include an unnecessary warning, legitimate-site blocking, user inconvenience, or extra analyst work.
- **False negative:** A phishing URL was incorrectly classified as legitimate. Possible consequences include undetected phishing, credential theft, malware delivery, or compromised organizational access.

The source emphasizes **recall** because missed phishing URLs can pose serious risks.

### 4.5 Underfitting and overfitting

```python
train_accuracy = dt_model.score(X_train, y_train)
test_accuracy = dt_model.score(X_test, y_test)

print("Training accuracy:", train_accuracy)
print("Testing accuracy:", test_accuracy)
```

The uploaded output shows this code but does **not** supply the numerical training/testing accuracy values.

**Underfitting:** The model is too simple to capture important patterns. For example, a tree with `max_depth=1` may lack enough decision splits to separate legitimate and phishing URLs. Both training and testing performance can be low.

**Overfitting:** The model learns overly specific rules for the training set. A very deep tree may fit small groups or individual observations, producing extremely high training performance but noticeably lower test performance.

**Why use `max_depth=5`?** The source gives these reasons:

- Keeps the model manageable.
- Reduces complexity.
- Makes interpretation easier.
- Illustrates control over tree growth.
- Reduces the tendency toward an extremely large tree.

A depth limit of 5 is **not necessarily universally optimal**; it is the controlled setting for this course experiment.

---

## Part 5: Short Analysis

### Question 1. What is a decision tree?

A decision tree is a supervised machine-learning algorithm that makes predictions by repeatedly asking questions about feature values.

### Question 2. What does `max_depth=5` control?

`max_depth` controls how deep the decision tree is allowed to grow. In this course experiment, `max_depth=5` limits tree complexity. Without that limit, a decision tree may produce increasingly specific rules, learning details unique to the training observations rather than general patterns.

### Question 3. Why might a very deep decision tree overfit the training data?

A deep decision tree may create rules for small groups or individual observations, fitting the training data too specifically. These learned details may not generalize to previously unseen examples.

### Question 4. Which metric do you think is especially important for phishing detection?

The source lists four choices—**Accuracy**, **Precision**, **Recall**, and **F1-score**—but does not include a completed answer for Question 4. The earlier discussion specifically emphasizes recall for phishing detection.

---

## Conversion Notes

This document was reformatted from the supplied `homework5.txt` output. Its section order, model values, reported results, and analysis are preserved in readable Markdown, with redundant separator lines and Google Colab HTML/CSS/JavaScript table-display artifacts omitted. The original text ends with the isolated number `1086`, which is not identified as part of an answer or metric and is therefore not presented as a result. No external references or experimental results have been added.
