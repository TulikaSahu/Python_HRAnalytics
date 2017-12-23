
# coding: utf-8

# In[1]:

#Importing important libraries 
import numpy as np
import pandas as pd
import seaborn as sns


# In[2]:

# moving spreadsheet dataset to python dataframe
df = pd.read_csv('HR_comma_sep.csv')  

# dimentions of dataset
df.shape

# First few records or rows of dataset
df.head()


# In[3]:

# dataset columns
col_names=df.columns.tolist()
print(col_names)


# In[4]:

## Data Preprocessing 

# Checking missing values in data
df.info()

#Renaming columns sales to department and average_montly_hours to average_monthly_hours
df=df.rename(columns={'sales':'department', 'average_montly_hours':'average_monthly_hours'})


# In[5]:

# Analysis of data

# importing libraries
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
#for all the plots to be in line
get_ipython().magic('matplotlib inline')
#matplot.lib for plotting 
import matplotlib.pyplot as plt


# In[6]:

# Correlation among attributes
HRcorr=df.corr()


plt.figure(figsize=(8, 8))
                        
sns.heatmap(HRcorr, vmax=1, square=True,annot=True,cmap='YlGnBu')
plt.title('Correlation among attributes')
plt.show()


# In[7]:

# Bar-graph to visualise and analyze each attribute

plt.style.use(style = 'ggplot')
df.hist(bins=11,figsize=(10,10),grid=True,color='green')
plt.show()


# In[8]:

# Further understanding of employee data 

import itertools

categorical=['number_project','time_spend_company','Work_accident','left', 'promotion_last_5years','department','salary']
fig=plt.subplots(figsize=(15,25))
length=len(categorical)

# colors Types
color_types = ['#78C850','#F08030','#6890F0','#A8B820','#A8A878','#A040A0','#F8D030',  
                '#E0C068','#EE99AC','#C03028','#F85888','#B8A038','#705898','#98D8D8','#7038F8']

for i,j in itertools.zip_longest(categorical,range(length)): # itertools.zip_longest for to execute the longest loop
    plt.subplot(np.ceil(length/2),2,j+1,facecolor='w')
    plt.subplots_adjust(hspace=.5)
    sns.set(style="whitegrid")
    sns.countplot(x=i,data = df,palette=color_types)
    plt.xticks(rotation=90)
    plt.title("Total Employee")


# In[9]:

# Number of projects and satisfaction level 
sns.set(font_scale=1)
g = sns.FacetGrid(df, col="number_project", row="left", margin_titles=True)
g.map(plt.hist, "satisfaction_level",color="purple")
plt.show()


# In[10]:

# no of employee who left the company 
print("The Number of employee who left the company :",len(df[df['left']==1]))
print("The Number of employee who didn't left the company",len(df[df['left']==0]))
print("The proportion of employee who left",len(df[df['left']==1])/len(df))


# In[11]:

# Analyzing behaviour of employees in terms of who left or stayed under each attribute condition

categorical=['number_project','time_spend_company','Work_accident','promotion_last_5years','department','salary'] 
fig=plt.subplots(figsize=(20,15))  
length=len(categorical)  

# itertools.zip_longest we use to execute the longest loop
for i,j in itertools.zip_longest(categorical,range(length)): 
    plt.subplot(np.ceil(length/2),2,j+1)  
    plt.subplots_adjust(hspace=.5)  
    sns.countplot(x=i,data = df,hue="left",palette=color_types)  
    plt.xticks(rotation=90)  
    plt.title("Employees who Left")  


# In[12]:

# Analyzing behaviour of employees for each attribute in terms of proportions

# Lets Calcualte proportion for above same
fig=plt.subplots(figsize=(15,15))
for i,j in itertools.zip_longest(categorical,range(length)):
    Proportion_of_employee = df.groupby([i])['left'].agg(lambda x: (x==1).sum()).reset_index()# only counting the number who left 
    Proportion_of_employee1=df.groupby([i])['left'].count().reset_index() # Counting the total number 
    Proportion_of_employee2 = pd.merge(Proportion_of_employee,Proportion_of_employee1,on=i) # mergeing two data frames
    
    # Now we will calculate the % of employee who left category wise
    Proportion_of_employee2["Proportion"]=(Proportion_of_employee2['left_x']/Proportion_of_employee2['left_y'])*100 
    Proportion_of_employee2=Proportion_of_employee2.sort_values(by="Proportion",ascending=False).reset_index(drop=True)#sorting by percentage
    plt.subplot(np.ceil(length/2),2,j+1)
    plt.subplots_adjust(hspace=.5)
    sns.barplot(x=i,y='Proportion',data=Proportion_of_employee2,palette=color_types)
    plt.xticks(rotation=90)
    plt.title("Employee who left in Percentage")
    plt.ylabel('Percentage')


# In[13]:

# Proportion of emplpoyees getting promotions 

fig=plt.subplots(figsize=(15,15))
sns.set(style="whitegrid")
categorical=['number_project','time_spend_company','department','salary']
length=len(categorical)
for i,j in itertools.zip_longest(categorical,range(length)):
    Proportion_of_employee = df.groupby([i])['promotion_last_5years'].agg(lambda x: (x==1).sum()).reset_index()# only counting the number who left 
    Proportion_of_employee1=df.groupby([i])['promotion_last_5years'].count().reset_index() # Counting the total number 
    Proportion_of_employee2 = pd.merge(Proportion_of_employee,Proportion_of_employee1,on=i) # mergeing two data frames
    # Now we will calculate the % of employee who  category wise
    Proportion_of_employee2["Proportion"]=(Proportion_of_employee2['promotion_last_5years_x']/Proportion_of_employee2['promotion_last_5years_y'])*100 
    Proportion_of_employee2=Proportion_of_employee2.sort_values(by="Proportion",ascending=False).reset_index(drop=True)#sorting by percentage
    plt.subplot(np.ceil(length/2),2,j+1)
    plt.subplots_adjust(hspace=.3)
    sns.barplot(x=i,y='Proportion',data=Proportion_of_employee2, palette=color_types)
    plt.xticks(rotation=90)
    plt.title("Employees Promotion in Percentage")
    plt.ylabel('Percentage')
    


# In[14]:

# Bar-graph to show Salary-wise distribution for employee who Left
f, ax = plt.subplots(figsize=(15, 4))
sns.set(style="whitegrid")
sns.countplot(y="salary", hue='left', data=df,orient="v", palette="Set3",edgecolor=sns.color_palette("dark", 10)).set_title('Distribution for Employee Salary who Left ');


# In[15]:

#Hypothesis testing 
#Comparing the means of the satisfaction for employees who left job against the employee population satisfaction
 
employee_population = df['satisfaction_level'][df['left'] == 0].mean()
employee_left_satisfaction = df[df['left']==1]['satisfaction_level'].mean()
print( 'The mean satisfaction for the employee population who Stayed : ' + str(employee_population))
print( 'The mean satisfaction for employees who Left: ' + str(employee_left_satisfaction) )


# In[16]:

#Conducting the T-Test
import scipy.stats as stats
stats.ttest_1samp(a=  df[df['left']==1]['satisfaction_level'], # Sample of Employee satisfaction who left
                  popmean = employee_population)  # Employee Who stayed satisfaction mean


# In[17]:

#T-Test Quantile

degree_offreedom = len(df[df['left']==1])

# Left Quartile
LeftQuartile = stats.t.ppf(0.025,degree_offreedom)  

# Right Quartile
RightQuartile = stats.t.ppf(0.975,degree_offreedom)  

print ('The t-distribution left quartile range is: ' + str(LeftQuartile))
print ('The t-distribution right quartile range is: ' + str(RightQuartile))


# In[18]:

## Percentage of employees who left or stayed per number of project 

fig=plt.subplots(figsize=(10,10))
ax = sns.barplot(x="number_project", y="number_project", hue="left", data=df,palette="Set3",edgecolor=sns.color_palette("dark", 25), estimator=lambda x: len(x) / len(df) * 100)
ax.set(ylabel="Percent")


# In[19]:


# Kernel Density Plot : Employee Evaluation Distribution
fig = plt.figure(figsize=(15,4),)
sns.set(style="whitegrid")
ax=sns.kdeplot(df.loc[(df['left'] == 0),'last_evaluation'] , color='g',shade=False,label='Stayed')
ax=sns.kdeplot(df.loc[(df['left'] == 1),'last_evaluation'] , color='r',shade=True, label='Left')
ax.set(xlabel='Employee Evaluation', ylabel='Frequency')
plt.title('Employee Evaluation Distribution - Left V.S Stayed')


#KDEPlot: Kernel Density Estimate Plot: Distribution for Employee AverageMonthly Hours
fig = plt.figure(figsize=(15,4))
ax=sns.kdeplot(df.loc[(df['left'] == 0),'average_monthly_hours'] , color='b',shade=False, label='Stayed')
ax=sns.kdeplot(df.loc[(df['left'] == 1),'average_monthly_hours'] , color='r',shade=True, label='Left')
ax.set(xlabel='Average Monthly Hours of Employee ', ylabel='Frequency')
plt.title('Distribution for Employee AverageMonthly Hours  - Left V.S. Stayed')


# In[20]:

## Data Transformation

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import RobustScaler

# Convert these categorical variables into numerical values
df["department"] = df["department"].astype('category').cat.codes
df["salary"] = df["salary"].astype('category').cat.codes
df.head()


# In[21]:

## Data Transformation
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap

y = np.ravel(df.loc[:,['left']])

# converting the object values of department and salary 
ind = list(enumerate(np.unique(df['department'])))
ind_dict = {name:i for i, name in ind}
df.sales = df.department.map(lambda x: ind_dict[x]).astype(int)

# same for salary
sal = list(enumerate(np.unique(df['salary'])))
sal_dict = {name:i for i, name in sal}
df.salary = df.salary.map(lambda x: sal_dict[x]).astype(int)
df.drop(['left'], axis=1 , inplace=True)

print(df.dtypes)


# In[22]:

#Standardizing the dataset
X = df.values
X_scaled = StandardScaler().fit_transform(X)
print(X_scaled.shape)


# In[23]:

## Dimentionality reduction by PCA
# PCA vis
from sklearn.decomposition import PCA
pcav = PCA().fit(X_scaled)
plt.plot(np.cumsum(pcav.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')

# PCA 
from sklearn.decomposition import PCA
pca = PCA(n_components=7).fit_transform(X_scaled)
print(pca.shape)


# In[24]:

# Splitting dataset into test and train (70:30)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=7, test_size=0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
data_train, data_test, Y_train, Y_test = train_test_split(pca, y, random_state=7, test_size=0.3)
print(data_train.shape, data_test.shape, Y_train.shape, Y_test.shape)

# Printing dimensions of data_train, data_test, y_train, y_test
print('data_train', data_train.shape)
print('data_test', data_test.shape)
print('Y_train', Y_train.shape)
print('Y_test', Y_test.shape)

# Printing dimensions ofX_train, X_test, y_train, y_test

print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)


# In[25]:

## Data modeling and algorithms 


# Importing Libraries for Random Forest and SVM
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef

# SVM Classification 

#### SVM without PCA 
svc_withoutpca = SVC(C=1, kernel='rbf', gamma=0.15).fit(X_train,y_train)
y_predict_SVM_withoutpca = svc_withoutpca.predict(X_test)

#Accuracy of SVM without PCA 
print('accuracy of SVM without PCA =',accuracy_score(y_test, y_predict_SVM_withoutpca))
a_SVM_withoutpca = accuracy_score(y_test, y_predict_SVM_withoutpca)


#Matthews correlation coefficient for SVM without PCA
m_SVM_withoutpca = matthews_corrcoef(y_test,y_predict_SVM_withoutpca)
print('Mathew\'s correlation coefficient for SVM without PCA =', m_SVM_withoutpca)



## SVM Classification with PCA
svc = SVC(C=1, kernel='rbf', gamma=0.15).fit(data_train,Y_train)
y_predict_SVM = svc.predict(data_test)

#Accuracy of SVM with PCA
print('accuracy of SVM with PCA =',accuracy_score(Y_test, y_predict_SVM))
a_SVM = accuracy_score(Y_test, y_predict_SVM)

#Matthews correlation coefficient for SVM with PCA
m_SVM = matthews_corrcoef(Y_test,y_predict_SVM)
print('Mathew\'s correlation coefficient for SVM with PCA =', m_SVM)


# In[26]:

# Random Forest Classification algorithm without PCA

clf_withoutpca = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1).fit(X_train, y_train)
y_predict_RF_withoutpca = clf_withoutpca.predict(X_test)
print('Feature importance by Ramdom forest classifier without PCA',clf_withoutpca.feature_importances_)

#Accuracy of Ramdom forest classifier
print('accuracy by Random Forest classifier without PCA=',accuracy_score(y_test, y_predict_RF_withoutpca))
a_RF_withoutpca = accuracy_score(y_test, y_predict_RF_withoutpca)

#Matthews correlation coefficient RF
m_RF_withoutpca = matthews_corrcoef(y_test,y_predict_RF_withoutpca)
print('Mathew\'s correlation coefficient for Random forest classifier without PCA =', m_RF_withoutpca)


# Random Forest Classification algorithm with PCA

clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1).fit(data_train, Y_train)
y_predict_RF = clf.predict(data_test)
print('Feature importance by Ramdom forest classifier ',clf.feature_importances_)

#Accuracy of Ramdom forest classifier
print('accuracy by Random Forest classifier =',accuracy_score(Y_test, y_predict_RF))
a_RF = accuracy_score(Y_test, y_predict_RF)

#Matthews correlation coefficient RF
m_RF = matthews_corrcoef(Y_test,y_predict_RF)
print('Mathew\'s correlation coefficient for Random forest classifier =', m_RF)


# In[27]:

#Feature importance details and visualization (without PCA)

important = clf_withoutpca.feature_importances_
indices = np.argsort(important)[::-1]
labels = df.columns
for i in range(X_train.shape[1]):
    print(i+1,important[indices[i]], labels[i])    
plt.title('feature importances')
plt.bar(range(X_train.shape[1]), important[indices], color='blue')
plt.xticks(range(X_train.shape[1]), labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()


# In[28]:

#Naive bayes Classification algorithm without PCA

from sklearn.naive_bayes import GaussianNB
classifier_withoutpca=GaussianNB()
classifier_withoutpca.fit(X_train,y_train)

#Predicting the test results
y_predict_NB_withoutpca = classifier_withoutpca.predict(X_test)

#Accuracy of Naive bayes classifier
print('accuracy by Naive Bayes classifier without PCA =', accuracy_score(y_test, y_predict_NB_withoutpca))
a_NB_withoutpca = accuracy_score(y_test, y_predict_NB_withoutpca)

#Mathews correlation coefficient for Naive bayes classifier
m_NB_withoutpca = matthews_corrcoef(y_test,y_predict_NB_withoutpca)
print('Mathew\'s correlation coefficient for Naive Bayes classifier without PCA =', m_NB_withoutpca)


###
#Naive bayes Classification algorithm with PCA

from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(data_train,Y_train)

#Predicting the test results
y_predict_NB = classifier.predict(data_test)

#Accuracy of Naive bayes classifier
print('accuracy by Naive Bayes classifier with PCA =',accuracy_score(Y_test, y_predict_NB))
a_NB = accuracy_score(Y_test, y_predict_NB)

#Mathews correlation coefficient for Naive bayes classifier
m_NB = matthews_corrcoef(Y_test,y_predict_NB)
print('Mathew\'s correlation coefficient for Naive Bayes classifier with PCA =',m_NB)


# In[29]:

#Decision Tree Classifier without PCA _withoutpca
from sklearn.tree import DecisionTreeClassifier as dt
classifier_withoutpca = dt(criterion='entropy',random_state=0)

classifier_withoutpca.fit(X_train,y_train)   

y_predict_DT_withoutpca = classifier_withoutpca.predict(X_test)

#Accuracy of Decision Tree Classifier
a_DT_withoutpca = accuracy_score(y_test, y_predict_DT_withoutpca)
print('accuracy by Decision Tree without PCA=',accuracy_score(y_test, y_predict_DT_withoutpca))

#Matthews correlation coefficient for Decision Tree Classifier
m_DT_withoutpca = matthews_corrcoef(y_test, y_predict_DT_withoutpca)
print('Matthew\'s correlation coefficient for Decision Tree Classifier without PCA =', m_DT_withoutpca)


###
#Decision Tree Classifier with PCA
from sklearn.tree import DecisionTreeClassifier as dt
classifier = dt(criterion='entropy',random_state=0)

classifier.fit(data_train,Y_train)   

y_predict_DT=classifier.predict(data_test)

#Accuracy of Decision Tree Classifier
a_DT = accuracy_score(Y_test, y_predict_DT)
print('accuracy by Decision Tree with PCA=',accuracy_score(Y_test, y_predict_DT))

#Matthews correlation coefficient for Decision Tree Classifier
m_DT = matthews_corrcoef(Y_test, y_predict_DT)
print('Matthew\'s correlation coefficient for Decision Tree Classifier with PCA=',m_DT)


#Decision Tree visulaization without PCA
import pydot
import pydotplus as pydot
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn import tree
from sklearn.tree import export_graphviz
 
dot_data = StringIO()

export_graphviz(classifier_withoutpca, out_file=dot_data, feature_names=labels)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.set_lwidth(400)
graph.set_lheight(300)

Image(graph.create_png())


# In[30]:

#KNN Classification algorithm
from sklearn.neighbors import KNeighborsClassifier

# When k= 3
classifier = KNeighborsClassifier(n_neighbors = 3, metric = 'minkowski',p=2)
classifier.fit(X_train, y_train)

#Predicting the results when k= 3
y_predict_KNNthree = classifier.predict(X_test)
a_KNNthree = accuracy_score(y_test, y_predict_KNNthree)
print('accuracy by 3-NN =',accuracy_score(y_test, y_predict_KNNthree))

m_KNNthree = matthews_corrcoef(y_test,y_predict_KNNthree)
print(m_KNNthree)


# When k= 5 without PCA
classifier_withoutpca = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p=2)
classifier_withoutpca.fit(X_train, y_train)

#Predicting the results when k= 5 
y_predict_KNN_withoutpca = classifier_withoutpca.predict(X_test)
a_KNN_withoutpca = accuracy_score(y_test, y_predict_KNN_withoutpca)
print('accuracy by 5-KNN without PCA =',accuracy_score(y_test, y_predict_KNN_withoutpca))

m_KNN_withoutpca = matthews_corrcoef(y_test,y_predict_KNN_withoutpca)
print(m_KNN_withoutpca)


# When k= 5 with PCA
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski',p=2)
classifier.fit(data_train, Y_train)

#Predicting the results when k= 5 
y_predict_KNN = classifier.predict(data_test)
a_KNN = accuracy_score(Y_test, y_predict_KNN)
print('accuracy by 5-KNN with PCA =',accuracy_score(Y_test, y_predict_KNN))

m_KNN = matthews_corrcoef(Y_test,y_predict_KNN)
print(m_KNN)


# When k= 7
classifier = KNeighborsClassifier(n_neighbors = 7, metric = 'minkowski',p=2)
classifier.fit(X_train, y_train)

#Predicting the results when k= 3
y_predict_KNNseven = classifier.predict(X_test)
a_KNNseven = accuracy_score(y_test, y_predict_KNNseven)
print('accuracy by 7-KNN =',accuracy_score(y_test, y_predict_KNNseven))

m_KNNseven = matthews_corrcoef(y_test,y_predict_KNNseven)
print(m_KNNseven)

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('K=3', 'K=5', 'K=7')
y_pos = np.arange(len(objects))
performance_withoutpca = [a_KNNthree,a_KNN_withoutpca,a_KNNseven]
 
plt.bar(y_pos, performance_withoutpca, align='center', alpha=0.7, width=0.3)
plt.xticks(y_pos, objects, fontsize=8)
plt.ylabel('Accuracy withoutpca')
plt.title('Comparision of accuracy with different K values in KNN classification without PCA')

plt.grid(True)
plt.show()


# In[31]:

# Comparision of accuracy for different classification models

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('Naive Bayes', 'Decision Tree', 'Random Forest', 'KNN', 'SVM')
y_pos = np.arange(len(objects))
performance = [a_NB,a_DT,a_RF,a_KNN,a_SVM]
 
plt.bar(y_pos, performance, align='center', alpha=0.7, width=0.3)
plt.xticks(y_pos, objects, fontsize=8)
plt.ylabel('Accuracy')
plt.title('Comparision of accuracy for different classification models')

plt.grid(True)
plt.show()


# In[32]:

# Comparision of accuracy for different classification models (considering RF & DT without PCA) 

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('Naive Bayes', 'Decision Tree', 'Random Forest', 'KNN', 'SVM')
y_pos = np.arange(len(objects))
performance = [a_NB,a_DT_withoutpca,a_RF_withoutpca,a_KNN,a_SVM]
 
plt.bar(y_pos, performance, align='center', alpha=0.7, width=0.3)
plt.xticks(y_pos, objects, fontsize=8)
plt.ylabel('Accuracy')
plt.title('Comparision of accuracy for different classification models (considering RF & DT without PCA)')

plt.grid(True)
plt.show()


# In[33]:

# Comparision of accuracy for RF and DT classification models when done without PCA and with PCA
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('Random Forest without PCA', 'Random Forest PCA', 'Decision Tree without PCA', 'Decision Tree PCA')
y_pos = np.arange(len(objects))
performance = [a_RF_withoutpca, a_RF, a_DT_withoutpca, a_DT]  
 
plt.bar(y_pos, performance, align='center', alpha=0.7, width=0.3)
plt.xticks(y_pos, objects, fontsize=8)
plt.ylabel('Accuracy')
plt.title('Comparision of accuracy for RF and DT classification models when done without PCA and with PCA')

plt.grid(True)
plt.show()


# In[34]:

# Comparision of Matthews correlation coefficient for different classification models

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('Naive Bayes', 'Decision Tree', 'Random Forest', 'KNN', 'SVM')
y_pos = np.arange(len(objects))
performance = [m_NB,m_DT,m_RF,m_KNN,m_SVM]
 
plt.bar(y_pos, performance, align='center', alpha=0.7, width=0.3)
plt.xticks(y_pos, objects, fontsize=8)
plt.ylabel('Accuracy')
plt.title('Comparision of Matthews correlation coefficient for different classification models')

plt.grid(True)
plt.show()


# In[35]:

# Comparision of Matthews correlation coefficient for different classification models (considering RF & DT without PCA)

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('Naive Bayes', 'Decision Tree', 'Random Forest', 'KNN', 'SVM')
y_pos = np.arange(len(objects))
performance = [m_NB,a_DT_withoutpca,m_RF_withoutpca,m_KNN,m_SVM]
 
plt.bar(y_pos, performance, align='center', alpha=0.7, width=0.3)
plt.xticks(y_pos, objects, fontsize=8)
plt.ylabel('Accuracy')
plt.title('Comparision of Matthews correlation coefficient for different classification models (considering RF & DT without PCA)')

plt.grid(True)
plt.show()


# In[ ]:



