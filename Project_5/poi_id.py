#!/usr/bin/python

import sys
import csv
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import tester
import enron
import matplotlib.pyplot as plt


from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'bonus',
                 'deferral_payments',
                 'deferred_income',
                 'director_fees',
                 'exercised_stock_options',
                 'expenses',
                 'loan_advances',
                 'long_term_incentive',
                 'other',
                 'restricted_stock',
                 'restricted_stock_deferred',
                 'salary',
                 'total_payments',
                 'total_stock_value',
                 'from_messages',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'shared_receipt_with_poi',
                 'to_messages']
 # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
### look at data:
'''
def make_csv(data_dict):
    """ generates a csv file from a data set"""
    fieldnames = ['name'] + data_dict.itervalues().next().keys()
    with open('data.csv', 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in data_dict:
            person = data_dict[record]
            person['name'] = record
            writer.writerow(person)

# ### 1.1 Dataset Exploration
print('# Exploratory Data Analysis #')
data_dict.keys()
print('Total number of data points: %d' % len(data_dict.keys()))
num_poi = 0
for name in data_dict.keys():
    if data_dict[name]['poi'] == True:
        num_poi += 1
print('Number of Persons of Interest: %d' % num_poi)
print('Number of people without Person of Interest label: %d' % (len(data_dict.keys()) - num_poi))

###1.2 Feature Exploration

all_features = data_dict["TOTAL"].keys()
print('Each person has %d features available' %  len(all_features))

### Evaluate dataset for completeness
missing_values = {}
for feature in all_features:
    missing_values[feature] = 0
for person in data_dict.keys():
    records = 0
    for feature in all_features:
        if data_dict[person][feature] == 'NaN':
            missing_values[feature] += 1
        else:
            records += 1
'''
# ### Print results of completeness analysis
# print('Number of Missing Values for Each Feature:')
# for feature in all_features:
#     print("%s: %d" % (feature, missing_values[feature]))


### Task 2: Remove outliers
def PlotOutlier(data_dict, feature_x, feature_y):
    """ Plot with flag = True in Red """
    data = featureFormat(data_dict, [feature_x, feature_y, 'poi'])
    for point in data:
        x = point[0]
        y = point[1]
        poi = point[2]
        if poi:
            color = 'red'
        else:
            color = 'blue'
        plt.scatter(x, y, color=color)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

### 2.1 Visualise outliers
'''
print(PlotOutlier(data_dict, 'salary', 'bonus'))

for k, v in data_dict.items():
    if v['salary'] != 'NaN' and v['salary'] > 10000000: 
    	print "outlier:"
    	print k
'''
###Remove outlier TOTAL line in pickle file.
data_dict.pop( 'TOTAL', 0 )
###Not a individual
data_dict.pop( 'THE TRAVEL AGENCY IN THE PARK', 0 )
###Only with NaN
data_dict.pop( 'LOCKHART EUGENE E:', 0 )
#print "Data removed."
'''
print(PlotOutlier(data_dict, 'salary', 'bonus'))

for k, v in data_dict.items():
    if v['salary'] != 'NaN' and v['salary'] > 1000000 and v['bonus'] != 'NaN' and v['bonus'] > 0.5: 
    	print "outlier:"
    	print k
'''


### Task 3: Create new feature(s)
enron.fraction_poi_communication(data_dict)
enron.total_wealth(data_dict)
### Store to my_dataset for easy export below.
features_list += ['fraction_poi_communication', 'total_wealth']
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

best_10_features = enron.get_k_best(data_dict, features_list, 10)
#print best_10_features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

def tune_logistic_regression():
    skb = SelectKBest()
    pca = PCA()
    lr_clf = LogisticRegression()

    pipe_lr = Pipeline(steps=[("SKB", skb), ("PCA", pca), ("LogisticRegression", lr_clf)])

    lr_k = {"SKB__k": range(9, 10)}
    lr_params = {'LogisticRegression__C': [1e-08, 1e-07, 1e-06],
                 'LogisticRegression__tol': [1e-2, 1e-3, 1e-4],
                 'LogisticRegression__penalty': ['l1', 'l2'],
                 'LogisticRegression__random_state': [42, 46, 50]}
    lr_pca = {"PCA__n_components": range(3, 8), "PCA__whiten": [True, False]}

    lr_k.update(lr_params)
    lr_k.update(lr_pca)

    enron.get_best_parameters_reports(pipe_lr, lr_k, features, labels)


def tune_random_forest():
    skb = SelectKBest()
    rf_clf = RandomForestClassifier()

    pipe_rf = Pipeline(steps=[("SKB", skb), ("RandomForestClassifier", rf_clf)])

    rf_k = {"SKB__k": range(8, 10)}
    rf_params = {'RandomForestClassifier__max_depth': [None, 5, 10],
                  'RandomForestClassifier__n_estimators': [10, 15, 20, 25],
                  'RandomForestClassifier__random_state': [42, 46, 50]}

    rf_k.update(rf_params)

    enron.get_best_parameters_reports(pipe_rf, rf_k, features, labels)


def tune_svc():
    skb = SelectKBest()
    pca = PCA()
    svc_clf = SVC()

    pipe_svc = Pipeline(steps=[("SKB", skb), ("PCA", pca), ("SVC", svc_clf)])

    svc_k = {"SKB__k": range(8, 10)}
    svc_params = {'SVC__C': [1000], 'SVC__gamma': [0.001], 'SVC__kernel': ['rbf']}
    svc_pca = {"PCA__n_components": range(3, 8), "PCA__whiten": [True, False]}

    svc_k.update(svc_params)
    svc_k.update(svc_pca)

    enron.get_best_parameters_reports(pipe_svc, svc_k, features, labels)


def tune_decision_tree():
    skb = SelectKBest()
    pca = PCA()
    dt_clf = DecisionTreeClassifier()

    pipe = Pipeline(steps=[("SKB", skb), ("PCA", pca), ("DecisionTreeClassifier", dt_clf)])

    dt_k = {"SKB__k": range(8, 10)}
    dt_params = {"DecisionTreeClassifier__min_samples_leaf": [2, 6, 10, 14],
                 "DecisionTreeClassifier__min_samples_split": [2, 6, 10, 14],
                 "DecisionTreeClassifier__criterion": ["entropy", "gini"],
                 "DecisionTreeClassifier__max_depth": [None, 5],
                 "DecisionTreeClassifier__random_state": [42, 46, 50]}
    dt_pca = {"PCA__n_components": range(3, 8), "PCA__whiten": [True, False]}

    dt_k.update(dt_params)
    dt_k.update(dt_pca)

    enron.get_best_parameters_reports(pipe, dt_k, features, labels)


### Naive Bayes:

clf_nb = GaussianNB()
print "GaussianNB : \n", tester.test_classifier(clf_nb, my_dataset, ['poi'] + best_10_features.keys())

#tune_logistic_regression()

### Logistic Regression 
eatures_lr = ['poi'] + enron.get_k_best(my_dataset, features_list, 9).keys()
clf_lr = Pipeline(steps = [('scaler', StandardScaler()), ('pca', PCA(n_components = 4, whiten = False)), 
     ('classifier', LogisticRegression(tol = 0.01, C = 1e-08, penalty = 'l2', random_state = 42))])

print "Logistic Regression : \n", tester.test_classifier(clf_lr, my_dataset, features_lr)


#tune_random_forest()
###Random Forest
'''
features_rf = ['poi'] + enron.get_k_best(my_dataset, features_list, 8).keys()
clf_rf = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', RandomForestClassifier(max_depth=5,
                                              n_estimators=10,
                                              random_state=46))])

print "Random Forest : \n", tester.test_classifier(clf_rf, my_dataset, features_rf)
'''
#tune_svc()
### SVC
'''
features_svc = ['poi'] + enron.get_k_best(my_dataset, features_list, 8).keys()
clf_svc = Pipeline(steps = [('scaler', StandardScaler()), ('pca', PCA(n_components = 5, whiten = True)), 
     ('classifier', SVC(C = 1000, gamma = 0.001, kernel = 'rbf'))])

print "Support vector classifier : \n", tester.test_classifier(clf_svc, my_dataset, features_svc)
'''

#tune_decision_tree()
###Decision Tree
'''
features_dt = ['poi'] + enron.get_k_best(my_dataset, features_list, 8).keys()

clf_dt = Pipeline(steps=[('scaler', StandardScaler()), ('pca', PCA(n_components=7, whiten=True)),
        ('classifier', DecisionTreeClassifier(criterion='entropy', min_samples_leaf=2, min_samples_split=6, random_state=46, max_depth=None))])

print "Decision Tree Classifier : \n",tester.test_classifier(clf_dt, my_dataset, features_dt)
'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_lr, my_dataset, features_lr)
