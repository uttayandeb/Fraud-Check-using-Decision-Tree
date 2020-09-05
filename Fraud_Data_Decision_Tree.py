##############################################################################
##################            Decision Tree               ####################




#Use decision trees to prepare a model on fraud data 
#treating those who have taxable_income <= 30000 as "Risky" and others are "Good"





#### Importing packages and loading dataset ############
import pandas as pd
import matplotlib.pyplot as plt
Fraud_Data = pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Desicion_tree\\Fraud_check.csv")
Fraud_Data.head()



Fraud_Data.isnull().any()#to check if we have null valuesin dataset or not
#so there are no null values in the dataset


Fraud_Data.dtypes# to check the data types


Fraud_Data.describe()### to check the summary of the dataset

Fraud_Data.info()


Fraud_Data['Undergrad'],class_names = pd.factorize(Fraud_Data['Undergrad'])
Fraud_Data.Undergrad
print(class_names)

Fraud_Data['Marital.Status'],class_names2=pd.factorize(Fraud_Data['Marital.Status'])
Fraud_Data['Marital.Status']
print(class_names2)

Fraud_Data['Urban'],class_names3=pd.factorize(Fraud_Data['Urban'])
Fraud_Data['Urban']
print(class_names3)



Fraud_Data.info()

Fraud_Data['Taxable.Income'].plot.hist()
plt.show()

Fraud_Data['Taxable.Income'].describe()

#count      600.000000
#mean     55208.375000
#std      26204.827597
#min      10003.000000
#25%      32871.500000
#50%      55074.500000
#75%      78611.750000
#max      99619.000000


#Converting the Sales column which is continuous into categorical
category = pd.cut(Fraud_Data['Taxable.Income'],bins=[0,30000,99619],labels=['Risky','Good'])
Fraud_Data.insert(0,'Taxable_Group',category)

Fraud_Data





import seaborn as sns
sns.pairplot(Fraud_Data)


Fraud_Data['Taxable_Group'].unique()
Fraud_Data.Taxable_Group.value_counts()#Good     476
                                       #Risky    124



######### Features Selection ##########


colnames = list(Fraud_Data.columns)
colnames
predictors = colnames[1:]#excluding 1st column all
predictors        #['Undergrad',
                  #'Marital.Status',
                  #'Taxable.Income',
                  #'City.Population',
                  #'Work.Experience',
                  #'Urban']feature variables

target = colnames[0]#only 1st column
target            #'Taxable_Group'          target variable


########## Splitting data into training and testing data set #############

import numpy as np

# np.random.uniform(start,stop,size) will generate array of real numbers with size = size


#### another way to split data into train and test

from sklearn.model_selection import train_test_split
train,test = train_test_split(Fraud_Data,test_size = 0.2)
test
test.shape # (120, 7)
train
train.shape #(480, 7)







##############  Decision Tree Model building ###############

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)

model = DecisionTreeClassifier(criterion = 'entropy')

model.fit(train[predictors],train[target])

preds = model.predict(test[predictors])

preds

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
print("Accuracy:",metrics.accuracy_score(test[target], preds))
#Accuracy: 0.9916666666666667


pd.Series(preds).value_counts()
#moderate    43
#Good     93
#Risky    27
#dtype: int64

pd.crosstab(test[target],preds)
#col_0          Good  Risky
#Taxable_Group             
#Risky             0     26
#Good             93      1

# Accuracy = train 
np.mean(train.Taxable_Group == model.predict(train[predictors]))#1.0

# Accuracy = Test
np.mean(preds==test.Taxable_Group) #  0.9916666666666667
 
model.score(test[predictors],test[target])








from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  #:pip install --upgrade scikit-learn==0.23.1
from IPython.display import Image  
import pydotplus
import io



dot_data = StringIO()
export_graphviz(model, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = predictors,class_names=['Good','Risky'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Fraud.png')
Image(graph.create_png())






