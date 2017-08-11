from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import r2_score
iris = datasets.load_iris()
classifier = linear_model.LinearRegression()
classifier.fit(iris.data,iris.target)
pred = classifier.predict(iris.data)
print "Accuracy = ",r2_score(iris.target,pred)