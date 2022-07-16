# ML model building 

# Import pandas
import pandas as pd
 
# reading csv file
df = pd.read_csv("/content/Positv_negativ_merged.csv")
print(df)

# To view the dataset distribution
import matplotlib.pyplot as plt
count_classes = pd.value_counts(df['Label'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Number of Protein targets in two classes")

plt.xticks(range(2))

plt.xlabel("Target classes")

plt.ylabel("Count")
#plt.legend("0 - Antibiotic ", "1 - Non Antibiotic")
#plt.legend(["0 - Antibiotic", "1 - Non Antibiotic"], loc ="lower right")
#plt.legend2("1- Non antibiotic")


# Drop the columns thats not required for training the model
x = df.drop('Pocket', axis=1)
x
X = x.drop('Label', axis=1)
X
y = df['Label']
y
print(X.shape, y.shape)


# To check if there's any column that's which has no importance and if any can be removed.
from sklearn.feature_selection import VarianceThreshold

def remove_low_variance(input_data, threshold=0.1):
    selection = VarianceThreshold(threshold)
    selection.fit(input_data)
    return input_data[input_data.columns[selection.get_support(indices=True)]]

X = remove_low_variance(X, threshold=0.1)
X


# Data Split(80/20):
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#X_train, X_test, y_train, y_test = train_test_split(X_train, y_test, test_size=0.2, random_state=42)

X_train.shape, X_test.shape


# RF Model building :
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef

model = RandomForestClassifier(max_depth=70, min_samples_leaf=3, min_samples_split=19, n_estimators=500, random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print (f'Train Accuracy - : {model.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {model.score(X_test,y_test):.3f}')

mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_train

mcc_test = matthews_corrcoef(y_test, y_test_pred)
mcc_test

# Import needed packages for classification report
import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(classification_report(y_test, y_test_pred))

#CV-5

from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=500, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
cv_scores

mcc_cv = cv_scores.mean()
mcc_cv

#CV-10

from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=500, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=10)
cv_scores

mcc_cv = cv_scores.mean()
mcc_cv

model_name = pd.Series(['Random forest'], name='Name')
mcc_train_series = pd.Series(mcc_train, name='MCC_train')
mcc_cv_series = pd.Series(mcc_cv, name='MCC_cv')
mcc_test_series = pd.Series(mcc_test, name='MCC_test')

performance_metrics = pd.concat([model_name, mcc_train_series, mcc_cv_series, mcc_test_series], axis=1)
performance_metrics

#SVM - (Linear, RBF, Polynomial) kernels
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
import numpy as np
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC(kernel='rbf', C=0.1).fit(X_train, y_train)
#classifier = svm.SVC(kernel='linear', C=0.1).fit(X_train, y_train)
#classifier = svm.SVC(kernel='poly', C=0.1).fit(X_train, y_train)
result = classifier.predict(X_test)
Score = metrics.accuracy_score(y_test, result)
print(Score)

np.set_printoptions(precision=2)

disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues)
title = "Confusion_Matrix"
disp.ax_.set_title(title)

print(title)
print(disp.confusion_matrix)

plt.show()

print (f'Train Accuracy - : {model.score(X_train,y_train):.3f}')
#print (f'Test Accuracy - : {model.score(X_test,y_test):.3f}')

from sklearn.metrics import classification_report
print(classification_report(y_test, result))

mcc_test = matthews_corrcoef(y_test, result)
mcc_test

mcc_train = matthews_corrcoef(y_train, y_train_pred)
mcc_train

#CV-5

from sklearn.model_selection import cross_val_score
classifier = svm.SVC(kernel='rbf', C=0.1).fit(X_train, y_train)
cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
cv_scores

mcc_cv = cv_scores.mean()
mcc_cv

#CV-10

from sklearn.model_selection import cross_val_score
classifier = svm.SVC(kernel='rbf', C=0.1).fit(X_train, y_train)
cv_scores = cross_val_score(classifier, X_train, y_train, cv=10)
cv_scores

mcc_cv = cv_scores.mean()
mcc_cv

model_name = pd.Series(['SVM'], name='Name')
mcc_train_series = pd.Series(mcc_train, name='MCC_train')
mcc_cv_series = pd.Series(mcc_cv, name='MCC_cv')
mcc_test_series = pd.Series(mcc_test, name='MCC_test')

performance_metrics = pd.concat([model_name, mcc_train_series, mcc_cv_series, mcc_test_series], axis=1)
performance_metrics


#XGBoost
import pandas as pd
import xgboost as xgb
#from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = xgb.XGBRegressor(
    n_estimators=500,
    reg_lambda=1,
    gamma=0,
    max_depth=70
)

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
mean_squared_error(y_test, y_pred)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

mcc_test = matthews_corrcoef(y_test, y_pred)
mcc_test

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print (f'Train Accuracy - : {model.score(X_train,y_train):.3f}')

#CV-5
from sklearn.model_selection import cross_val_score
classifier = regressor.fit(X_train, y_train)
cv_scores = cross_val_score(classifier, X_train, y_train, cv=5)
cv_scores
mcc_cv = cv_scores.mean()
mcc_cv

#CV-10
from sklearn.model_selection import cross_val_score
classifier = regressor.fit(X_train, y_train)
cv_scores = cross_val_score(classifier, X_train, y_train, cv=10)
cv_scores
mcc_cv = cv_scores.mean()
mcc_cv

model_name = pd.Series(['SVM'], name='Name')
mcc_train_series = pd.Series(mcc_train, name='MCC_train')
mcc_cv_series = pd.Series(mcc_cv, name='MCC_cv')
mcc_test_series = pd.Series(mcc_test, name='MCC_test')

performance_metrics = pd.concat([model_name, mcc_train_series, mcc_cv_series, mcc_test_series], axis=1)
performance_metrics


#ROC - CURVE

import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=.25,
                                                    random_state=1234)


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score

# Instantiate the classfiers and make a list
classifiers = [LogisticRegression(random_state=1234), 
               GaussianNB(), 
               KNeighborsClassifier(), 
               XGBClassifier(),
               #SVC(random_state=1234),
               svm.SVC(kernel='linear', probability=True),
               GradientBoostingClassifier(random_state=1234),
               DecisionTreeClassifier(random_state=1234),
               RandomForestClassifier(random_state=1234)]

# Define a result table as a DataFrame
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

# Train the models and record the results
for cls in classifiers:
    model = cls.fit(X_train, y_train)
    #lr = SVC(gamma='auto', probability=True)
    yproba = model.predict_proba(X_test)[::,1]

    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':cls.__class__.__name__,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()


#Ensemble Model(Customizable)

from collections import Counter
counter=Counter(y)
counter

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline

# get models
# get a voting ensemble of models
# define the base models
models = list()

decision_tree = Pipeline([('m', DecisionTreeClassifier())])
models.append(('decision', decision_tree))

randomforest = Pipeline([('m', RandomForestClassifier())])
models.append(('randomforest', randomforest))

svc = Pipeline([('m', SVC())])
models.append(('svc', svc))

# define the voting ensemble
ensemble = VotingClassifier(estimators=models, voting='hard')

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(ensemble, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

n_scores
n_scores.mean()