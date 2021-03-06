from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


# load the iris dataset form sklearn
digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=.75)

clf = RandomForestClassifier(n_estimators=10,
                             max_depth=15,
                             min_samples_leaf=8,
                             max_features=30,
                             n_jobs=-1
                             )
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

train_conf_mtx = confusion_matrix(y_train, y_train_pred)
test_conf_mtx = confusion_matrix(y_test, y_test_pred)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print("")
print("scikit-learn")
print("")
print("Training")
print(train_conf_mtx)
print("Train Accuracy:", train_acc)
print("")
print("Testing")
print(test_conf_mtx)
print("Test Accuracy:", test_acc)
