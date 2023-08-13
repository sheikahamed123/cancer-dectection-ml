import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import tree

print("Dataset:")
dataset=pd.read_csv('lung_cancer_examples.csv')
print(len(dataset))
print(dataset.head())

scatter_matrix(dataset)
pyplot.show()

A=dataset[dataset.Result==1]
B=dataset[dataset.Result==0]

plt.scatter(A.Age, A.Smokes, color="Black", label="1", alpha=0.4)
plt.scatter(B.Age, B.Smokes, color="Blue", label="0", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Smokes")
plt.legend()
plt.title("Smokes vs Age")
plt.show()

plt.scatter(A.Age, A.Alkhol, color="Black", label="1", alpha=0.4)
plt.scatter(B.Age, B.Alkhol, color="Blue", label="0", alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Alkhol")
plt.legend()
plt.title("Alkhol vs Age")
plt.show()

plt.scatter(A.Smokes, A.Alkhol, color="Black", label="1", alpha=0.4)
plt.scatter(B.Smokes, B.Alkhol, color="Blue", label="0", alpha=0.4)
plt.xlabel("Smokes")
plt.ylabel("Alkhol")
plt.legend()
plt.title("Smokes vs Alkhol")
plt.show()

x=dataset.iloc[:, 3:5]
y=dataset.iloc[:,6]
x_train, x_test, y_train, y_test=train_test_split(x, y, random_state=0, test_size=0.2)

#feature scaling

sc_x= StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)


print("---------------------USING KNN ALGORITHM----------------------")
import math
a=math.sqrt(len(y_train))
print(a)

#defining a model- knn
classifier=KNeighborsClassifier(n_neighbors=5, p=2, metric='euclidean')

#fit model
classifier.fit(x_train,y_train)

#predict the test set result

y_pred=classifier.predict(x_test)
print(y_pred)

#evaluate model
#confusion matrix
cm=confusion_matrix(y_test,y_pred)
print("confusion matrix")
print(cm)
print("In confusion matrix-----")
print("Position 1.1 shows the position that don't have cancer, in this case = 8")
print("Position 1.2 shows the number of patients that have higher risk of cancer, in this case = 1")
print("Position 2.1 shows the incorrect value,in this case = 2")
print("Position 2.2 shows the correct number of patients that have cancer, in this case =2")

print("F1 Score : ", (f1_score(y_test, y_pred))*100)

print('ACCURACY : ', (accuracy_score(y_test, y_pred))*100)

#using decision tree

print("--------------------USING DECISION TREE ALGORITHM-------------------")
c=tree.DecisionTreeClassifier()
c.fit(x_train, y_train)
accu_train = np.sum(c.predict(x_train)==y_train)/float(y_train.size)
accu_test = np.sum(c.predict(x_test)==y_test)/float(y_test.size)
print("Clssification accuracy on train", (accu_train)*100)
print("Classification accuracy on test", (accu_test)*100)