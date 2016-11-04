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
# def make_csv(data_dict):
#     """ generates a csv file from a data set"""
#     fieldnames = ['name'] + data_dict.itervalues().next().keys()
#     with open('data.csv', 'w') as csvfile:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for record in data_dict:
#             person = data_dict[record]
#             person['name'] = record
#             writer.writerow(person)

# ### 1.1 Dataset Exploration
# print('# Exploratory Data Analysis #')
# data_dict.keys()
# print('Total number of data points: %d' % len(data_dict.keys()))
# num_poi = 0
# for name in data_dict.keys():
#     if data_dict[name]['poi'] == True:
#         num_poi += 1
# print('Number of Persons of Interest: %d' % num_poi)
# print('Number of people without Person of Interest label: %d' % (len(data_dict.keys()) - num_poi))


# ###1.2 Feature Exploration
# all_features = data_dict["TOTAL"].keys()
# print('Each person has %d features available' %  len(all_features))
# ### Evaluate dataset for completeness
# missing_values = {}
# for feature in all_features:
#     missing_values[feature] = 0
# for person in data_dict.keys():
#     records = 0
#     for feature in all_features:
#         if data_dict[person][feature] == 'NaN':
#             missing_values[feature] += 1
#         else:
#             records += 1

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

# 2.1 Visualise outliers

# print(PlotOutlier(data_dict, 'salary', 'bonus'))

# for k, v in data_dict.items():
#     if v['salary'] != 'NaN' and v['salary'] > 10000000: 
#     	print "outlier:"
#     	print k

#Remove outlier TOTAL line in pickle file.
data_dict.pop( 'TOTAL', 0 )
#Not a individual
data_dict.pop( 'THE TRAVEL AGENCY IN THE PARK', 0 )
#Only with NaN
data_dict.pop( 'LOCKHART EUGENE E:', 0 )
print "Data removed."

# print(PlotOutlier(data_dict, 'salary', 'bonus'))

# for k, v in data_dict.items():
#     if v['salary'] != 'NaN' and v['salary'] > 1000000 and v['bonus'] != 'NaN' and v['bonus'] > 0.5: 
#     	print "outlier:"
#     	print k



### Task 3: Create new feature(s)
enron.fraction_poi_communication(data_dict)
enron.total_wealth(data_dict)
### Store to my_dataset for easy export below.
features_list += ['fraction_poi_communication', 'total_wealth']
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

best_10_features = enron.get_k_best(data_dict, features_list, 9)

print best_10_features

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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

dump_classifier_and_data(clf, my_dataset, features_list)