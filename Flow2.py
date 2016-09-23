from sklearn import svm
from sklearn import datasets
from sklearn.svm import SVC
import pickle

clf = svm.SVC()

iris = datasets.load_iris()

print "Data"
print "----"
x = iris.data
print x

print "Target"
print "------"
y = iris.target
print y

clf.fit(x, y)
SVC(C=1.0, cache_size=200, class_wt=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto',
    kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)

s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print clf2.predict(x[0:1])




