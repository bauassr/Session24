# Core Libraries - Data manipulation and analysis
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Core Libraries - Machine Learning, Preprocessing and generating Performance Metrics
import sklearn
from sklearn import preprocessing
from sklearn import metrics

# Importing Classifiers - Modelling
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

## Importing train_test_split,cross_val_score,GridSearchCV,KFold - Validation and Optimization
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit


 # Loading the data into the dataframe
url= 'https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv'
titanic = pd.read_csv(url)   

#Get Understading of Dataset Titanic
print("Shape of dataset  titanic\n",titanic.shape)
print("\nTop 5 values of dataset \n",titanic.head())
print("\n\n Lets check info of Titanic dataset \n")
print(titanic.info(),"\n As we can see clear there null values in few columns")

#Data Cleaning 
print("\n\nPclass, Sex, Age, SibSp (Siblings aboard), Parch (Parents/children aboard), and Fare to predict whether a passenger survived.")
#drop columns
titanic.drop(axis =1, columns= ["PassengerId","Name","Ticket","Cabin","Embarked"], inplace = True)
titanic.head()

print("\n \n Check null values from  different columns\n")
print(titanic.isna().sum())


print("\n Lets replace NaN with mean for Age\n")
# Taking care of missing data
titanic['Age'].fillna((titanic['Age'].mean()), inplace=True)
print(titanic.isna().sum())


print("\n Selecting categorical columns")
# Selecting categorical columns to feature engineer
cat_cols = titanic.select_dtypes(include='object').columns.values
print(cat_cols)

print("\n\nEncoding the Sex columns values into 0 and 1 and creating a new column with those values\n")
# Encoding the Sex columns values into 0 and 1 and creating a new column with those values
titanic['Sex'] = titanic['Sex'].replace({'female':0, 'male': 1})
print(titanic.head())
print("\n Now all columns have only numerical values")

print("\n Lets Now Get X and Y variables for 'o'b Modeling \n")
X = titanic.drop("Survived", axis = 1);
Y = titanic.Survived
print("\n X TOP 5\n\n",X.head(),"\n\n Y Top 5\n\n",Y.head())
print("\n\n Let get correlation b/w diffrent columns  \n")
plt.figure(figsize=(10,10))
sns.heatmap(titanic.corr(), annot = True)


print("\n\n SLPIT TEST AND TRAIN DATASET \n\n")
x_train,x_test,y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state =100)

print("\n\n BUILD MODEL\n\n")
lr =  LogisticRegression()
lr.fit(x_train, y_train)
y_test_pred= lr.predict(x_test)

print("Logistic Regression Classifier - Base",
      "\n\t Accuracy:", metrics.accuracy_score(y_test, y_test_pred),
      "\n\t Precision:", metrics.precision_score(y_test, y_test_pred),
      "\n\t Recall:", metrics.recall_score(y_test, y_test_pred),
      "\n\t Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_test_pred),
      "\n\t Classification Report:\n",  metrics.classification_report(y_test, y_test_pred),"\n") 


cart =  DecisionTreeClassifier()
cart.fit(x_train, y_train)
y_test_pred= cart.predict(x_test)

print("Decision Tree Classifier - Base",
      "\n\t Accuracy:", metrics.accuracy_score(y_test, y_test_pred),
      "\n\t Precision:", metrics.precision_score(y_test, y_test_pred),
      "\n\t Recall:", metrics.recall_score(y_test, y_test_pred),
      "\n\t Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_test_pred),
      "\n\t Classification Report:\n",  metrics.classification_report(y_test, y_test_pred),"\n")   

# Plotting graph visual of the descision tree
dot_data = tree.export_graphviz(cart, out_file=None, filled=True, rounded=True,
                                feature_names=['Pclass', 'Sex_Encoded', 'Age', 'SibSp', 'Parch', 'Fare'])

# Importing Graphing and Visualization tools
import pydotplus
from IPython.display import Image
graph = pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))

cart.get_params

print("\n\n Hyper parameter optimization ")
# Initializing the classifier to optimize, 
# Setting CV split and tree hyper-parameters for using in GridSearchCV optimization

cart_classifier =  DecisionTreeClassifier()

CV = ShuffleSplit(test_size=0.20, random_state=100)

param_grid = {  
              'criterion':['gini','entropy'], 
              'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
              'max_features':[2,3,4,5,6], 
              'max_leaf_nodes': [2, 3, 4, 6, 9],
              'min_impurity_decrease': np.linspace(0.1,0.5,4),
              'min_samples_leaf':[ 2, 3, 5, 7],
              'min_samples_split':[2, 3, 5], 
              'random_state' : [100]
            }
rscv_grid = GridSearchCV(cart_classifier, param_grid=param_grid, verbose=1)
rscv_grid.fit(x_train, y_train)
# Showing the best hyper-parameters for the decision tree
rscv_grid.best_params_
# Using the best estimator created from the above hyper-parameters listed in the params_grid
model = rscv_grid.best_estimator_
model.fit(x_train, y_train)
y_pred_test = model.predict(x_test)
print("Decision Tree Classifier - Best Estimator",
      "\n\t Accuracy:", metrics.accuracy_score(y_test, y_pred_test),
      "\n\t Precision:", metrics.precision_score(y_test, y_pred_test),
      "\n\t Recall:", metrics.recall_score(y_test, y_pred_test),
      "\n\t Confusion Matrix:\n", metrics.confusion_matrix(y_test, y_pred_test),
      "\n\t Classification Report:\n",  metrics.classification_report(y_test, y_pred_test),"\n")  
dot_data = tree.export_graphviz(rscv_grid.best_estimator_, out_file=None, filled=True, rounded=True,
                                feature_names=['Pclass', 'Sex_Encoded', 'Age', 'SibSp', 'Parch', 'Fare'])
graph = pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))                 
