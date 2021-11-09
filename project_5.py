#  librires 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_theme(style="darkgrid")

#modify the display options to view entire dataframe
pd.options.display.max_columns = None


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

orginal_train = train.copy()
orginal_test = test.copy()

# understanding our data by small steps and visual tools
def understand_our_data(data):
    print('Our Data Info','\n')
    print(data.info(),'\n')
    print('Describe our Numeric data','\n')
    print(data.describe(),'\n')
    print('Describe our Objectiv data','\n')
    print(data.describe(include=['O']),'\n')
    print('Objects columns','\n')
    print(data.dtypes == 'object','\n')
    print('Sorted type of columns','\n')
    print(data.dtypes.sort_values(),'\n')
    print('Number of null values','\n')
    print(data.isna().sum().sort_values(),'\n')
    print('Shape of our Data','\n')
    print(data.shape,'\n')
    print('Percnt of test data comparing of train data','\n')
    print(test.shape[0]/data.shape[0],'\n')
    print('Number of unique vales','\n')
    print(data.nunique().sort_values(),'\n')
    print('percantge of null values', '\n')
    print(round(data.isna().sum(axis=0)/len(data),2)*100)


np.array(train.columns)

num_train = train.select_dtypes(include= np.number)
obj_train = train.select_dtypes(include= 'object')

# check our data
understand_our_data(num_train)
understand_our_data(obj_train)


# visulization our  numaeric data
def plot_hist_boxplot(column, train):
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    sns.distplot(train[train[column].notnull()][column], ax=ax[0])
    sns.boxplot(train[train[column].notnull()][column], ax=ax[1])
#    print('skewness for {} : '.format(column),skew(train[train[column].notnull()][column]))
#    print('kurtosis for {} : '.format(column), kurtosis(train[train[column].notnull()][column]))
    plt.show()

for column in num_train.columns:
    plot_hist_boxplot(column, train)

# study category data

def category(columns, data):
    category_data = []
    for col in columns:
        category_data.append(data[col].value_counts())
    return (category_data)

category_1 = category(obj_train.columns, obj_train)
len_category =list(map(lambda x: len(x), category_1))

category_count = {}

for val in len_category:
    if val in category_count.keys():
        count = category_count[val]
        category_count[val] = count+1
    else:
        category_count[val] = 1

#plot showing the count of columns having same number of unique values
keys = category_count.keys()
values = category_count.values()
plt.bar(keys, values,width=0.8, color='red')
plt.xlabel('Distinct Values in Categorical Variable', fontsize=15)
plt.ylabel('Count', fontsize=15)
plt.title('Categorical Labels with Same Unique Values',fontsize=20)
plt.rcParams['figure.figsize'] = [48/2.54, 10/2.54]
plt.show()

# find category column        
column_datatypes = train.dtypes
categorical_columns = list(column_datatypes[column_datatypes=="object"].index.values)
'''
 we have 12 cont column [1-14], id and loss non of them has null values 

'''
# deal with missing values and outliers

from sklearn.impute import SimpleImputer

class Data_preprocessing:
    def __init__(self, train):
        self.train = train
        
    def missing_cont(self, column_names_with_specific_type, imputation_type='mean'):
        if imputation_type=='mean':
            mean_imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
            mean_imputer.fit(self.train[column_names_with_specific_type])
            self.train[column_names_with_specific_type] = mean_imputer.transform(
                self.train[column_names_with_specific_type])
            
        if imputation_type=='median':
            median_imputer = SimpleImputer(missing_values = np.nan, strategy='median')
            median_imputer.fit(self.train[column_names_with_specific_type])
            self.train[column_names_with_specific_type] = median_imputer.transform(
                self.train[column_names_with_specific_type])
            
        return self.train
    
    def missing_obj(self, column_names_with_specific_type, imputation_type='most_frequent'):
        if imputation_type=='most_frequent':
            most_frequent = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
            most_frequent.fit(self.train[column_names_with_specific_type])
            self.train[column_names_with_specific_type] = most_frequent.transform(
                self.train[column_names_with_specific_type])
    
    def outlier_tratemnt(self, Q1, Q3, IQR, columns_with_outlier, action):
        if action=='median':
            for i in range(len(columns_with_outlier)):
                column_name = columns_with_outlier[i]
                meadian_outlier = np.median(self.train[column_name])
                self.train.loc[self.train[((self.train[column_name]<(Q1[column_name]-(1.5*IQR[column_name])))|(self.train[column_name]>(Q3[column_name]+(1.5*IQR[column_name]))))].index,column_name]=meadian_outlier
                
        if action=='mean':
            for i in range(len(columns_with_outlier)):
                column_name = columns_with_outlier[i]
                mean_outlier = np.mean(self.train[column_name])
                self.train.loc[self.train[((self.train[column_name]<(Q1[column_name]-(1.5*IQR[column_name])))|(self.train[column_name]>(Q3[column_name]+(1.5*IQR[column_name]))))].index,column_name]=mean_outlier   
            
        if action=='remove':
            for i in range(len(columns_with_outlier)):
                column_name = columns_with_outlier[i]
                self.train = self.train[~((self.train_data[column_name]<(Q1[column_name]-(1.5*IQR[column_name])))|(self.train_data[column_name]>(Q3[column_name]+(1.5*IQR[column_name]))))]
        
        return self.train
        
# apply our class for missing values
data_preproccesing = Data_preprocessing(train)
train = data_preproccesing.missing_cont(num_train.columns.tolist(), 'median')
train = data_preproccesing.missing_obj(obj_train.columns.tolist())

# detect our outliers
num_train_1 = num_train.drop(['id', 'loss'], axis =1)
ax = sns.boxplot(data=num_train_1, orient='h', palette='Set2' )

# apply our calss for outliers
columns_with_outlier = ['cont7', 'cont9', 'cont10']
Q1 = num_train_1.quantile(0.25)
Q3 = num_train_1.quantile(0.75)
IQR =(Q3 - Q1)

train = data_preproccesing.outlier_tratemnt(Q1, Q3, IQR, columns_with_outlier, 'median')

num_train_1= train.select_dtypes(include= np.number)
num_train_1 = num_train_1.drop(['id', 'loss'], axis =1)
ax = sns.boxplot(data=num_train_1, orient='h', palette='Set2')

#Function for feature selection of numeric variables
#Remove variables with constant variance
#Remove variables with Quasi-Constant variance with a fixed threshold
#Remove correlated variables
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder


def feature_selection_numerical_variables(train, qthrshold, corr_threshold,exclude_numerical_cols_list):
    num_col = train.select_dtypes(include=np.number).columns.tolist()
    num_col = [col for col in num_col if col not in exclude_numerical_cols_list]
    
    #remove variables with constant variance
    constant_filter = VarianceThreshold(threshold=0)
    constant_filter.fit(train[num_col])
    constant_col = [column for column in num_col if column not in
                    train[num_col].columns[constant_filter.get_support()]]
    if len(constant_col)>0:
        train.drop(labels=contsant_col, axis=1, inplace=True)
        
    
    #remove deleted columns from dataframe
    num_col = [column for column in num_col if column not in constant_col]
    
    #remove variables with qconstant variance
    #Remove quasi-constant variables
    qconstant_filter = VarianceThreshold(threshold=qthrshold)
    qconstant_filter.fit(train[num_col])
    qconstant_columns = [column for column in num_col if column not in
                    train[num_col].columns[constant_filter.get_support()]]
    if len(qconstant_columns)>0:
        train.drop(labels=qconstant_columns, axis=1, inplace=True)
    
    #remove deleted columns from dataframe
    num_col = [column for column in num_col if column not in qconstant_columns]
    
    #remove correlated variables
    correlated_features = set()
    correlation_matrix = train[num_col].corr()
    ax= sns.heatmap(correlation_matrix, vmin=-1, vmax=1, center=0, square=True,
                    cmap=sns.diverging_palette(20, 220, n=200))
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')
    #print(correlation_matrix)
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i,j])>corr_threshold:
                colname = correlation_matrix.columns[i]
                colcompared = correlation_matrix.columns[j]
                #check if the column compared against is not in the columns excluded list
                if colcompared not in correlated_features:
                    correlated_features.add(colname)
    train.drop(labels=correlated_features, axis=1, inplace=True)
    
    return train,constant_col,qconstant_columns,correlated_features

train,constant_columns,qconstant_columns,correlated_features =feature_selection_numerical_variables(train,0.01,0.75,['loss','id'],)
   
#Visualizing the distribution of loss value
# Density Plot and Histogram of loss
sns.distplot(train['loss'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

#We will use a log transformation on the dependent variable to reduce the scale
train['loss'] = np.log(train['loss'])

# Visualizing the distribution of loss value
# Density Plot and Histogram of loss
sns.distplot(train['loss'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

# convert category to binary 
from sklearn.preprocessing import LabelEncoder
for cf1 in categorical_columns:
    le = LabelEncoder()
    le.fit(train[cf1].unique())
    train[cf1] = le.transform(train[cf1])
    
# count unique vales

count_unique = pd.DataFrame(columns =['column', 'count'])
for col in categorical_columns:
    unique_values = len(train[col].unique())
    count_unique = count_unique.append({'column': col, 'count': int(unique_values)}, ignore_index=True)


    
# insert model
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

#convert the int64 columns categorical
Column_datatypes= train.dtypes
Integer_columns = list(Column_datatypes.where(lambda x: x =="int64").dropna().index.values)
train[Integer_columns] = train[Integer_columns].astype('category',copy=False)
X,y = train.drop(['id','loss'],axis=1),train['loss']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Instantiate model with 100 decision trees
rf_base = RandomForestRegressor(n_estimators = 100, random_state = 42,oob_score = True)
rf_base.fit(X_train, y_train)

#validate the accuracy of the base model
#compare the model accuracies
Y_test_predict_base = rf_base.predict(X_test)
print("Base model accuracy:",np.sqrt(mean_squared_error(y_test, Y_test_predict_base)))

# Create the random grid
random_grid = {'n_estimators': [100,200,300,400,500],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(10,110, num=11)],
               'min_samples_split': [200,400,600],
               'min_samples_leaf': [1,2,4],
               'bootstrap': ['True', 'False']}

# Use the random grid to search for best hyperparameters
# base model to tune
rf = RandomForestRegressor()

# 5 fold cross validation, 
# search across 150 different combinations, and use all available cores
rf_tuned = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, cv = 3,n_iter = 5, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_tuned.fit(X_train, y_train)
rf_tuned.best_params_

Y_test_predict_tuned = tunedmodel_rf.predict(X_test)
print("Tuned model accuracy:",np.sqrt(mean_squared_error(y_test, Y_test_predict_tuned)))

# try Gradientboostingregressor
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
gbm_base = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=3,
    learning_rate=1.0)

gbm_base.fit(X_train,y_train)
gbm_base.best_params_
Y_test_predict_tuned = gbm_base.predict(X_test)
print("Base model GBM accuracy:",np.sqrt(mean_squared_error(y_test, Y_test_predict_tuned)))        













