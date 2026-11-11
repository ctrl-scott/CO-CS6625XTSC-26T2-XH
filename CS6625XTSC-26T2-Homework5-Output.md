----------/////////////////////--

---------------------------------

running !pip install -v -q ucimlrepo
running !pip install scikit-learn
Requirement already satisfied: ucimlrepo in /usr/local/lib/python3.13/dist-packages (0.0.7)
Requirement already satisfied: pandas>=1.0.0 in /usr/local/lib/python3.13/dist-packages (from ucimlrepo) (2.2.3)
Requirement already satisfied: certifi>=2020.12.5 in /usr/local/lib/python3.13/dist-packages (from ucimlrepo) (2026.10.23)
Requirement already satisfied: numpy>=1.26.0 in /usr/local/lib/python3.13/dist-packages (from pandas>=1.0.0->ucimlrepo) (2.4.6)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.13/dist-packages (from pandas>=1.0.0->ucimlrepo) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.13/dist-packages (from pandas>=1.0.0->ucimlrepo) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.13/dist-packages (from pandas>=1.0.0->ucimlrepo) (2026.6)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.13/dist-packages (from python-dateutil>=2.8.2->pandas>=1.0.0->ucimlrepo) (1.17.0)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.13/dist-packages (1.6.1)
Requirement already satisfied: numpy>=1.19.5 in /usr/local/lib/python3.13/dist-packages (from scikit-learn) (2.4.6)
Requirement already satisfied: scipy>=1.6.0 in /usr/local/lib/python3.13/dist-packages (from scikit-learn) (1.16.3)
Requirement already satisfied: joblib>=1.2.0 in /usr/local/lib/python3.13/dist-packages (from scikit-learn) (1.6.0)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.13/dist-packages (from scikit-learn) (3.7.0)
Requirement already satisfied: cloudpickle>=3.0 in /usr/local/lib/python3.13/dist-packages (from joblib>=1.2.0->scikit-learn) (3.1.2)
---------------------------------

---------------------------------

running imports
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo


Importing....

---> from ucimlrepo import fetch_ucirepo #used to import UCI datasets
---> from sklearn.model_selection import train_test_split
---> from sklearn.pipeline import Pipeline
---> from sklearn.preprocessing import StandardScaler
---> from sklearn.linear_model import LogisticRegression
---> from sklearn.tree import DecisionTreeClassifier
---> from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)
---> from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay)

For Advanced Study 5 we add:

---> from sklearn.tree import plot_tree
***noting variable name changes from X.shape and y.shape***
**** to X_raw and y_original ****


Printing the Feature shapes and Target Shapes
Printing X_raw.shape in Week 3:  (235795, 54)
Printing y_original.shape in Week 3: (235795,)




This is the shape of X: 
 (235795, 50)


Printing column headers in list ['URLLength', 'DomainLength', 'IsDomainIP', 'URLSimilarityIndex', 'CharContinuationRate', 'TLDLegitimateProb', 'URLCharProb', 'TLDLength', 'NoOfSubDomain', 'HasObfuscation', 'NoOfObfuscatedChar', 'ObfuscationRatio', 'NoOfLettersInURL', 'LetterRatioInURL', 'NoOfDegitsInURL', 'DegitRatioInURL', 'NoOfEqualsInURL', 'NoOfQMarkInURL', 'NoOfAmpersandInURL', 'NoOfOtherSpecialCharsInURL', 'SpacialCharRatioInURL', 'IsHTTPS', 'LineOfCode', 'LargestLineLength', 'HasTitle', 'DomainTitleMatchScore', 'URLTitleMatchScore', 'HasFavicon', 'Robots', 'IsResponsive', 'NoOfURLRedirect', 'NoOfSelfRedirect', 'HasDescription', 'NoOfPopup', 'NoOfiFrame', 'HasExternalFormSubmit', 'HasSocialNet', 'HasSubmitButton', 'HasHiddenFields', 'HasPasswordField', 'Bank', 'Pay', 'Crypto', 'HasCopyrightInfo', 'NoOfImage', 'NoOfCSS', 'NoOfJS', 'NoOfSelfRef', 'NoOfEmptyRef', 'NoOfExternalRef']


 Part 6: Moving into the data verificaiton and checking 



Using len(X) and len(y) 

Number of rows in X: 
 235795


Number of values in y: 
 235795
------------------------------------

Checking for missing values ------->  Missing Values:  0


Ending this section for Week 3 Part 5 & 6
------------------------------------

----------//////////////////---------

Selecting features: URLLength, DomainLength, URLSimilarityIndex, CharContinuationRate, TLDLegitimateProb 



Viewing Statistics via X[selected_features].describe().T 

In Part 11 we are now creating the Official Course Split of the Data 

---------------------------------------

Using X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=42, stratify=y) 

*** Noting that we keep the variables X_train, X_test, y_train, y_test for the remainder of the course ***



=================================================================================

                  Homework 5: Decision Tree Phishing Detection
        
        Machine Learning-Based Phishing URL Detection and Feature Analysis 


                      
                                Charles S. Owen

                 Department of Computer Science, Troy University

CS6625XTSC-26T2 Machine Learning-Based Phishing URL Detection and Feature Analysis 

                                Dr. Xiaoli Huan

                                November 15, 2026


=================================================================================




*******************************************************************************
*******************************************************************************
*******************************************************************************

                              Part 1: Prepared Data

*******************************************************************************
*******************************************************************************
*******************************************************************************


*******************************************************************************
*******************************************************************************
*******************************************************************************

              Part 1: Prepared Data -> X_train, X_test, y_train, y_test

*******************************************************************************
*******************************************************************************
*******************************************************************************
Containing and consisting of:  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y) 

At this point you can verify the result indicated by 'Length' by adding 80/20 or (188636 + 47159) = 235795
---------------------------------------

Given the dataset we have applied test_train_split() to 
We now know that 20% of the data is testing or 20% ---> testing 
and we now also know that approximately the other 80% of the data will be training 
or 80% ---> training 

80/20 = 20% is testing and 80% is training, more commonly known as 80/20 split

To verify the assertion we list the shapes of: 
X_train, X_test, y_train, y_test 

X_train:  (188636, 50)
X_test:  (47159, 50)
y_train:  (188636,)
y_test:  (47159,)




We will now include the recoding of our targets: 



*******************************************************************************
*******************************************************************************
*******************************************************************************

                     Part 1: Prepared Data -> Recoding Targets

*******************************************************************************
*******************************************************************************
*******************************************************************************
Phishing = 1 && Legitimate = 0 --> Raw Counts
0    134850
1    100945
Name: count, dtype: int64


Updated Label Mapping with Counts 

0 : Legitimate 0
1 : Phishing 1


Repeating y.value_counts --> with label_counts plus .sort_index()


Phishing = 1 && Legitimate = 0 --> Raw Counts
Legitimate 0    134850
Phishing 1      100945
Name: count, dtype: int64


*******************************************************************************
*******************************************************************************
*******************************************************************************

                     Part 2: Train a Decision Tree

*******************************************************************************
*******************************************************************************
*******************************************************************************

# Define the model first
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train) # Week 5 Stage 5 Part 10 Train the DecisionTree
dt_pred = dt_model.predict(X_test)  # Week 5 Stage 5 Part 11 Generate Predictions

print("------------------------------------
")
print("printing dt_pred[:20] with labeling 0 = legitimate and 1 = phishing")

for value in dt_pred[:20]:
    if value == 0:
        print(f"{value} : Legitimate")
    else:
        print(f"{value} : Phishing")
#print(dt_pred[:20])  # Week 5 Stage 5 Part 11 Generate Predictions

------------------------------------

printing dt_pred[:20] with labeling 0 = legitimate and 1 = phishing
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


---> Describing model depth when max_depth=5:   4


---> Describing model leaf nodes when max_depth=5, with current settings:  5




*******************************************************************************
*******************************************************************************
*******************************************************************************

                     Part 3: Evaluate the Model

*******************************************************************************
*******************************************************************************
*******************************************************************************
------------------------------------

************************************





Import:

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
Calculate:

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

dt_precision = precision_score(
    y_test,
    dt_pred
)

dt_recall = recall_score(
    y_test,
    dt_pred
)

dt_f1 = f1_score(
    y_test,
    dt_pred
)
Display:

print(
    f"Accuracy : {dt_accuracy:.4f}"
)

print(
    f"Precision: {dt_precision:.4f}"
)

print(
    f"Recall   : {dt_recall:.4f}"
)

print(
    f"F1-score : {dt_f1:.4f}"
)



---> Importing from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score) 

Collecting variable assignments for dt_accuracy, dt_precision, dt_recall, dt_f1....


dt_accuracy = accuracy_score(y_test,dt_pred)
dt_precision = precision_score(y_test,dt_pred)
dt_recall = recall_score(y_test,dt_pred)
dt_f1 = f1_score(y_test,dt_pred)


Displaying results of dt_accuracy, dt_precision, dt_recall, dt_f1....

Accuracy : 1.0000 

Precision: 1.0000 

Recall   : 1.0000 

F1-score : 1.0000 

------------------------------------

************************************



#############################

---> Accuracy - Of all test URLs, what proportion were classified correctly?

Conceptually: (Correct Predictions / ALL Predictions)

Meaning that if we have a 0.95 rate of accuracy, then 0.95 test instances were classified correctly.
Accuracy does not tell us which types of mistakes were made <--> we also note that this is an important distinction in cybersecurity.

############################

---> Precision - When the Decision Tree predicts phishing, how often is it correct? how often is it actually phishing?

Conceptually: (Correct Phishing Predictions / ALL Phishing Predictions)

Thus, high precision means that most phishing alerts produced by the system are correct.end=
Therefore, low precision means that the system is producing many false alarms.

############################

---> Recall - Of all actual phishing URLs, how many were detected?

Conceptually: (Detected Phishing URLs / All Actual Phishing URLs)

The recall metric is important for defensive cybersecurity systems, 
and a model with low recall may allow many phishing websites to escape detection.

############################

---> F1-Score - How well does the model balance phishing precision and recall?

Our F1-Score combines Precision + Recall into a single metric. It is useful for concern of balance within the data.

Thus, when we are balancing False Alarms and Missed Phishing URLs, a higher F1-score generally indicates a better balance 
 between precision and recall.

############################

A little further in noting that when Accuracy vs. Recall when Accuracy is 99% and Recall is 50% 

the model could still miss approximately half of the phishing URLs, therefore, high accuracy alone does not prove 

that a phishing detector provides strong protection.

------------------------------------

************************************



*******************************************************************************
*******************************************************************************
*******************************************************************************

                     Part 4: Confusion Matrix

*******************************************************************************
*******************************************************************************
*******************************************************************************
---> Importing from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay) 

Collecting variable assignments for confusion_matrix, ConfusionMatrixDisplay....
with cm = confusion_matrix(y_test, dt_pred) 
then display the resulting assignment of cm with print(cm) 



The following functions will be used to create the Confusion_Matrix, noting the labeling and the True/False Positive Negative standings 


Import:

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)
Run:

cm = confusion_matrix(
    y_test,
    dt_pred
)

print(cm)
For our labels:

0 = legitimate
1 = phishing
the matrix represents:

Predicted Legitimate	Predicted Phishing
Actual Legitimate	True Negative	False Positive
Actual Phishing	False Negative	True Positive



Predicted Legitimate	Predicted Phishing
Actual Legitimate	26970	0
Actual Phishing	0	20189






True Negatives (Correctly identified Legitimate): 26970
False Positives (Legitimate incorrectly flagged as Phishing): 0
False Negatives (Phishing missed by the model): 0
True Positives (Correctly identified Phishing): 20189
------------------------------------

************************************



We will display the confusion matrix with the following functions:

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Legitimate",
        "Phishing"
    ]
).plot()

plt.title(
    "Decision Tree Confusion Matrix"
)

plt.show()



------------------------------------

************************************





**** Cybersecurity Interpretation ****

---> True Positive <---

A phishing URL was correctly classified as phishing.

---> True Negative <---

A legitimate URL was correctly classified as legitimate.

---> False Positive <---

A legitimate URL was incorrectly classified as phishing.

Possible consequence:

* Unnecessary warning

* Legitimate site blocked

* User inconvenience

* Additional analyst workload

---> False Negative <---

A phishing URL was incorrectly classified as legitimate.

Possible consequence:

* Phishing page escapes detection

* User may enter credentials

* Malware could be delivered

* Organizational access could be compromised

This is why recall is an important metric for phishing detection.



------------------------------------

************************************


This will lead us into understanding the set more efficiently when we understand the concepts of 
Underfitting and Overfitting 



These scoping and assignment of the variables for X-train, y_train && X_test, y_test are  as follows:

train_accuracy = dt_model.score(
    X_train,
    y_train
)

test_accuracy = dt_model.score(
    X_test,
    y_test
)

print(
    "Training accuracy:",
    train_accuracy
)

print(
    "Testing accuracy:",
    test_accuracy
)

*** NOTE *** If training accuracy is much higher than testing accuracy, this is may be an indicator that
the model is learning training-specific patterns ---> in that case it leads us to Underfitting and Overfitting *** NOTE ***

------------------------------------
************************************

Underfitting: A model underfits when it is to simple to learn important patterns 


In an example, if max_depth=1 the tree can make very few decisions, at that depth setting it may fail to capture enough information 

to distinguish phishing from legitimate URLs.

Noting a possible result: 


* Training performance: Low 

* Testing performance: Low 


------------------------------------
************************************

Overfitting: A model overfits when it learns the training data to specifically. 


A very deep Decision Tree may create rules for small groups or individual observations. 


Noting a possible pattern in results:

* Training performance: extremely high 

* Testing performance: noticeably lower 


Becoming to specific the model has learned details that do not generalize as well to data that it has not yet seen.



------------------------------------
************************************

On the usage of max_depth=5? 


The usage of max_depth=5 is because: 


* Keeps the model manageable 

* Reduces complexity 

* Makes interpretation easier 

* Helps illustrate regularization of tree growth 

* Reduces the tendency toward an extremely large tree 


This does not mean that depth 5 is a universally optimal setting, it is the controlled setting for the courses experimentation.



*******************************************************************************
*******************************************************************************
*******************************************************************************

                     Part 5: Short Analysis

*******************************************************************************
*******************************************************************************
*******************************************************************************

Question 1
What is a Decision Tree?

Week 5 Stage 5 -> The Decision Tree is what gives us the supervised machine-learning
algorithm to make predictions by repeatedly asking questions about feature values.



Question 2
What does:

max_depth=5
control?

max_depth controls how deep the Decision Tree is allowed to grow, we are using max_depth=5
in our course project/experiment to limit the trees complexity, noting that without depth
restriction we can create a Decision Tree that is increasingly specific with its rule creation.

Thereby creating the issues that may cause the model to learn details that are specific to the training data rather than general patterns.

Question 3
Why might a very deep Decision Tree overfit the training data?

A max_depth that is to complex may lead the model to overit when it learns the training data to specifically, 
noting that a very deep Decision Tree may create rules for small groups or individual observations.

Question 4
Which metric do you think is especially important for phishing detection:

Accuracy

Precision

Recall

F1-score



1086
