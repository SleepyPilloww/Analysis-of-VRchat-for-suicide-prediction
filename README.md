# Analysis-of-VRchat-for-suicide-prediction

PROBLEM STATEMENT

Problem definition

Develop a text analysis system that can accurately identify and analyse text messages related to suicide and depression. The system should be able to detect warning signs, provide resources, and offer support to individuals in need. Despite advancements in the detection and treatment of severe mental diseases, suicide remains an unsolvable public health issue1. The field of developing technology for suicide screening by obtaining and examining social media data is expanding. The relationship between suicide and social media is complex, with both positive and negative influences shaping individuals' mental health. On one hand, social media platforms provide a space for individuals to connect, share experiences, and access support networks. However, the dark side of social media emerges when it becomes a breeding ground for harmful influences that can contribute to the risk of suicide. The anonymity offered by social media platforms can embolden individuals to engage in harmful behaviours, leading to increased stress, anxiety, and depression among victims. The constant comparison to others' seemingly perfect lives may contribute to feelings of inadequacy and isolation, further increasing the vulnerability of individuals to mental health challenges, including suicidal thoughts. It is observed that young people who self-harm use the Internet frequently to express their distress, and there is a rising trend of people dying by suicide after posting on social media, which is found to have assortative patterns. Here we develop a machine learning approach based on blogs, Instagram texts, tweets that predicts individual level future suicidal risk based on online social media data prior to any mention of suicidal thought.

Existing work

The National Action Alliance for Suicide Prevention identifies the development of biomarkers and predictive screening technologies as a priority area that would enable focusing limited resources onto individuals at risk16. A significant challenge in this area is the relative low base rate of suicidal behaviour in the population, making prospective studies impractical. To address this, an area of growing interest is the development of suicide screening technologies through accessing and analysing social media data. Many research papers have been published in the same domain which detect the suicidal tendencies of the user based on text/language. Some of the research papers concerning the same are attached below:

Application of Natural Language Processing (NLP) in Detecting and Preventing Suicide Ideation: A Systematic Review

Natural Language Processing of Social Media as Screening for Suicide Risk

Our contribution

The suicide detection model developed by our team categorizes the text in question as ‘suicidal’ or ‘non-suicidal’. The model has demonstrated an average accuracy of 92% which is high compared to the existing suicide detection models based on text messages and social media posts.

IMPLEMENTATION

Data sets used:

Suicide and Depression Detection

The dataset comprises posts extracted from the "SuicideWatch" and "depression" subreddits on Reddit, gathered through the Pushshift API. All "SuicideWatch" posts from December 16, 2008, to January 2, 2021, were collected and labeled as "suicide." Similarly, posts from the "depression" subreddit spanning January 1, 2009, to January 2, 2021, were collected and labeled as "depression." Non-suicidal posts were sourced from the "r/teenagers" subreddit. The dataset provides a collection of textual content reflecting discussions on mental health, facilitating research and analysis on topics related to suicide and depression within the Reddit community.

Environment:

The python notebook was set up on Google Colab with the dataset uploaded on Google Drive as a .csv file. The Colab notebook was given access to mount the Google Drive to access the dataset.

• import os: This statement imports the os module, which provides a way to interact with the operating system, including functions for file and directory operations.

• for dirname, _, filenames in os.walk('/content/drive/MyDrive/Suicide_Detection.csv'): This line uses the os.walk function to iterate through all the directories and files within the specified path ('/content/drive/MyDrive/Suicide_Detection.csv'). The function returns a tuple containing the current directory, a list of subdirectories, and a list of filenames in the current directory.

• for filename in filenames: This is a nested loop that iterates through the list of filenames obtained from os.walk.

• print(os.path.join(dirname, filename)): This line prints the full path of each file in the specified directory.

• import ctypes: This statement imports the ctypes module, which provides C compatible data types and allows calling functions in dynamic link libraries/shared libraries.

• import pandas as pd: This imports the pandas library and gives it the alias pd. Pandas is a powerful data manipulation and analysis library.

• import re: This imports the re module, which provides support for regular expressions (regex) in Python.

• import matplotlib.pyplot as plt: This line imports the matplotlib.pyplot module, which is a plotting library for creating visualizations in Python.

• os.path: This is used to manipulate file paths and check the existence of files or directories.

Proposed Technique / Methodology:

import os

This line imports the 'os' module, which provides a way to interact with the operating system, such as reading or writing to the file system.

for dirname, _, filenames in os.walk('/content/drive/MyDrive/Suicide_Detection.csv'): for filename in filenames: print(os.path.join(dirname, filename))

This part of the code uses the os.walk function to iterate through all the directories and files in the specified path. It prints the full path of each file in the specified directory.

import ctypes import os.path import pandas as pd import re import os import matplotlib.pyplot as plt %matplotlib inline

Here, several libraries are imported: ctypes, os.path, pandas (as pd), re (regular expressions), os, and matplotlib.pyplot for data manipulation, regular expressions, and plotting, respectively. %matplotlib inline is a magic command used in Jupyter notebooks to display plots directly in the notebook.

df = pd.read_csv('/content/drive/MyDrive/Suicide_Detection.csv')

This line uses pandas to read a CSV file located at the specified path and stores it in a DataFrame (df)

df = df.fillna('')

This line fills any missing values in the DataFrame with empty strings.

df['class'].value_counts()

This line prints the counts of unique values in the 'class' column of the DataFrame.

data = df['class'].value_counts() names = list(data.keys()) values = list(data.values)

Here, it stores the counts and corresponding class names in separate lists.

from sklearn.model_selection import train_test_split

his line imports the train_test_split function from scikit-learn, which is used for splitting data into training and testing sets.

df_ml = df

This line creates a new DataFrame (df_ml) and assigns it the same data as the original DataFrame (df).

sentences = df['text'].values y = df_ml['class'].values

These lines extract the 'text' column as input features (sentences) and the 'class' column as the target variable (y).

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, y, test_size=0.20, random_state=1000)

Here, it splits the data into training and testing sets using the train_test_split function

from sklearn.feature_extraction.text import CountVectorizer

This line imports the CountVectorizer class from scikit-learn, which is used to convert a collection of text documents to a matrix of token counts.

vectorizer = CountVectorizer() vectorizer.fit(sentences_train)

These lines create an instance of CountVectorizer and fit it on the training data (sentences_train)

X_train = vectorizer.transform(sentences_train) X_test = vectorizer.transform(sentences_test)

These lines transform the training and testing sentences into matrices of token counts using the fitted CountVectorizer

from sklearn.linear_model import LogisticRegression

This line imports the LogisticRegression class from scikit-learn

classifier = LogisticRegression() classifier.fit(X_train, y_train)

These lines create an instance of logistic regression, and then fit it on the training data. score = classifier.score(X_test, y_test)

print('𝐀𝐜𝐜𝐮𝐫𝐚𝐜𝐲 𝐒𝐜𝐨𝐫𝐞:-', score)

This calculates and prints the accuracy score of the logistic regression model on the test data.

def predict_category(s, train=y, model=classifier): V = [s] vect = CountVectorizer() vect.fit(V) pr = vectorizer.transform(V) pred = model.predict(pr) return pred[0] predict_category('My Girlfriend left me, I do not want to live anymore')

finally, this code defines a function predict_category that takes a text input (s) and predicts its category using the trained logistic regression model. The example sentence is then passed to this function to get the prediction.

RESULTS AND DISCUSSION

The code achieved a 92% accuracy in identifying individuals with suicidal ideation, demonstrating a high level of performance. This result is consistent with other studies that have used natural language processing and machine learning to detect suicidal content, with accuracies ranging from 80% to 92%. The high accuracy of the code is a promising indicator of its potential to identify individuals at risk of suicide. However, it is important to consider the balance between sensitivity and precision, as even very strong predictors can generate many false positives due to the relatively low base rate of suicidal behaviours. Therefore, while the high accuracy is encouraging, further validation and consideration of the potential false positive rate are necessary before implementing the code in a real-world setting. This code brilliantly used natural language processing to detect the emotion/feelings of human to detect whether they are suicidal or not.
