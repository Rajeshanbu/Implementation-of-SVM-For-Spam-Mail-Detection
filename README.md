# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect a labeled dataset of emails, distinguishing between spam and non-spam.
2. Preprocess the email data by removing unnecessary characters, converting to lowercase, removing stop words, and performing stemming or lemmatization.
3. Extract features from the preprocessed text using techniques like Bag-of-Words or TF-IDF.
4. Split the dataset into a training set and a test set.
5. Train an SVM model using the training set, selecting the appropriate kernel function and hyperparameters.
6. Evaluate the trained model using the test set, considering metrics such as accuracy, precision, recall, and F1 score.
7. Optimize the model's performance by tuning its hyperparameters through techniques like grid search or random search.
8. Deploy the trained and fine-tuned model for real-world use, integrating it into an email server or application.
9. Monitor the model's performance and periodically update it with new data or adjust hyperparameters as needed.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: RAJESH A
RegisterNumber: 212222100042
```
```py
import pandas as pd
data = pd.read_csv("spam.csv",encoding = "windows - 1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![SVM For Spam Mail Detection](sam.png)
### data.head()

![9 1](https://github.com/Rajeshanbu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118924713/f7e64e17-c5aa-4565-bc49-d58035b4af96)


### data.info()

![9 2](https://github.com/Rajeshanbu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118924713/63e4c25d-d935-465c-9210-5b59cc9cd264)


### data.isnull().sum()

![9 3](https://github.com/Rajeshanbu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118924713/f892a64d-f1f2-4130-9b68-2d957636169f)


### Y_prediction value

![9 4](https://github.com/Rajeshanbu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118924713/79dab771-ca0a-4f4b-97e3-9f43414640e9)


### Accuracy value

![9 5](https://github.com/Rajeshanbu/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118924713/bbd97f08-da31-4b3d-b269-20c82c29b5d5)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
