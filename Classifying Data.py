import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plot

# Loading the data
id_num = int(input("Enter ID of dataset: "))
data = fetch_ucirepo(id=id_num)

# data (as pandas dataframes)
x = data.data.features
y = data.data.targets
x.drop_duplicates()
x.dropna()
y.infer_objects(copy=False)

# Changing class values to numeric values (make it customizable)
while True:
    old = input("Enter class you want to replace: ")
    if old.isnumeric():
        old = int(old)
    new = input("Enter its replacement: ")
    if new.isnumeric():
        new = int(new)
    y.replace(old, new)
    answer = input("Replace another? [y/n]: ")
    if answer.lower() == 'n':
        break


# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Decision Tree
md = int(input("Enter max depth: "))
dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=md)
dt.fit(x_train, y_train.values.ravel())
test_results_dt = dt.predict(x_test)
accuracy_dt = accuracy_score(y_test, test_results_dt)
print(f'Accuracy for Decision Tree: {accuracy_dt * 100}%')
tree.plot_tree(dt, feature_names=x.columns.values)

# Nearest Neighbor
n = int(input("Enter number of neighbours: "))
knn = KNeighborsClassifier(n_neighbors=n)
knn.fit(x_train, y_train.values.ravel())
test_results_knn = knn.predict(x_test)
accuracy_knn = accuracy_score(y_test, test_results_knn)
print(f'Accuracy for KNN: {accuracy_knn * 100}%')

# Naive Bayes Theorem
nb = GaussianNB()
nb.fit(x_train, y_train.values.ravel())
test_results_nb = nb.predict(x_test)
accuracy_nb = accuracy_score(y_test, test_results_nb)
print(f'Accuracy for Naive Bayes: {accuracy_nb * 100}%')

# Support Vector Machine
SVM = SVC()
SVM.fit(x_train, y_train.values.ravel())
test_results_svm = SVM.predict(x_test)
accuracy_svm = accuracy_score(y_test, test_results_svm)
print(f'Accuracy for SVM: {accuracy_svm * 100}%')

# Showing the decision tree
plot.show()
