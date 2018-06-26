import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

##reading different views
df=pd.read_csv('/home/chetan/Desktop/my_projects/iris_dataset/iris.data.txt',names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class'])


##different views

#print(df)
#print(df.shape)
#print(df.head(10))
#print(df.describe())
#print(df.groupby('class').size())
#df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()
#df.hist()
#plt.show()
#scatter_matrix(df)
#plt.show()

#### steps for validation of dataset

array=df.values##returns values of df in  numpy format
X=array[:,0:4]#from 0 to 4
Y=array[:,4]#only 4
v_size=0.20
seed=7
x_train,x_valid,y_train,y_valid=model_selection.train_test_split(X,Y,test_size=v_size,random_state=seed)

###now building all models
seed = 7
scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
#plt.show()


###now using knn directly
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
predictions = knn.predict(x_valid)
print(accuracy_score(y_valid , predictions))
print(confusion_matrix(y_valid, predictions))
print(classification_report(y_valid , predictions))
