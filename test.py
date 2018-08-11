from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


# comparison of our RF and sklearn-RF using iris data
iris = load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


from RandomForests import RandomForests
my_forest = RandomForests(n_estimators=100, max_features=np.sqrt)
my_forest.fit(x_train, y_train)
y_predict = my_forest.predict(x_test)
print('The accuracy of my Random Forests is {}'.format(accuracy_score(y_predict, y_test)))


from sklearn.ensemble import RandomForestClassifier
sklearn_forest = RandomForestClassifier(n_estimators=100, max_features='sqrt')
sklearn_forest.fit(x_train, y_train)
y_predict2 = sklearn_forest.predict(x_test)
print('The accuracy of SKLearn Random Forests is {}'.format(accuracy_score(y_predict2, y_test)))

