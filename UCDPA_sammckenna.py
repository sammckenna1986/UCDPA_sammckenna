import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

def import_data():
    data = pd.read_csv("BankChurners.csv")
    # creating a dataframe
    df =pd.DataFrame(data, columns= ['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level' ,'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio'])
    return df

def exploring_data():
    df = import_data()
    # converting categorical values for numerical values here: Attrition_Flag 'Existing Customer' with '0' and 'Attrited Customer' with '1'
    df['Attrition_Flag'].replace({'Existing Customer': int(0), 'Attrited Customer': int(1)},
                                 inplace=True)  # https://www.kite.com/python/answers/how-to-replace-column-values-in-a-pandas-dataframe-in-python
    df['Gender'].replace({'M': int(0), 'F': int(1)}, inplace=True)
    # df['Card_Category'].replace({'Blue': int(1), 'Silver': int(2), 'Gold': int(3), 'Platinum': int(4)})

    # Printing the info and describe functions to have a more general look at the data
    print(df['Attrition_Flag'])
    print(df.describe())
    # checking for missing values
    print("Null values total:" + str(df.isnull().sum()))
    """
    if I had any null values I would use the information from the following website to drop 'the rows that included 
    null values and therefore might have been incomplete to use': https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
    """
    # Choosing the prediction target
    y = df['Attrition_Flag']

    # Selecting features -- I removed the Attrition Flag Column
    data_features = ['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
                     'Marital_Status', 'Income_Category', 'Card_Category', 'Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio']

    # Calling the multiple features X
    X = df[data_features]

    # Explore the data set a bit more and start to look at graphs for anomalies in the 'type of customers', 'how the data is setup', and 'what might be an indicator of the customers leaving'.
    print("The shape of X is:" + str(X.shape))  # results 10127 rows and 20 columns
    print("There are" + str(df.isnull().sum()) + "null values in the dataset")
    print("Describing X:" + str(X.describe()))
    print("The head of X:" + str(X.head()))
    # using the describe method to look at the data and see if there is anything interesting at a high level in the data.
    # The average dependent is 2.3 and the average months on the book is 35 months, average age is 46 years old, min so youngest is 26 etc...
    print(df.info())  # checking what all of the column data types are

    # I really like this from https://www.kaggle.com/jieyima/income-classification-model it allows you to see the unique values for the columns like the unique google sheets function
    print('Attrition_Flag', df[
        'Attrition_Flag'].unique())
    print('Customer_Age', df['Customer_Age'].unique())
    print('Gender', df['Gender'].unique())
    print('Dependent_count', df['Dependent_count'].unique())
    print('Education_Level', df['Education_Level'].unique())
    print('Marital_Status', df['Marital_Status'].unique())
    print('Income_Category', df['Income_Category'].unique())
    print('Card_Category', df['Card_Category'].unique())
    print('Months_on_book', df['Months_on_book'].unique())
    print('Total_Relationship_Count', df['Total_Relationship_Count'].unique())
    print('Months_Inactive_12_mon', df['Months_Inactive_12_mon'].unique())
    print('Contacts_Count_12_mon', df['Contacts_Count_12_mon'].unique())
    print('Credit_Limit', df['Credit_Limit'].unique())
    print('Total_Revolving_Bal', df['Total_Revolving_Bal'].unique())
    print('Avg_Open_To_Buy', df['Avg_Open_To_Buy'].unique())
    print('Total_Amt_Chng_Q4_Q1', df['Total_Amt_Chng_Q4_Q1'].unique())
    print('Total_Trans_Amt', df['Total_Trans_Amt'].unique())
    print('Total_Trans_Ct', df['Total_Trans_Ct'].unique())
    print('Total_Ct_Chng_Q4_Q1', df['Total_Ct_Chng_Q4_Q1'].unique())
    print('Avg_Utilization_Ratio', df['Avg_Utilization_Ratio'].unique())

def CHART_Months_on_BooksVS_attrition():
    df = import_data()
    sns.catplot(data=df, kind="count", x="Months_on_book", y=None,
                hue="Attrition_Flag", )  # https://seaborn.pydata.org/introduction.html https://seaborn.pydata.org/generated/seaborn.catplot.html
    plt.show()  # This was interesting as a lot of the customers seem to be there for around 36 months and it might show that the dataset might be skewed or they had a marketing campaign that attracted a lot of customers 36 months ago.

def CHART_ContactsCountVS_attrition():
    df = import_data()
    sns.catplot(data=df, kind="count", x="Contacts_Count_12_mon", y=None,
                hue="Attrition_Flag", )  # https://seaborn.pydata.org/introduction.html https://seaborn.pydata.org/generated/seaborn.catplot.html
    plt.show()

def CHART_TansactionsCountVS_attrition(): ####there is a problem here.
    df = import_data()
    sns.catplot(data=df, kind="count", x="Total_Trans_Ct", y=None,
                hue="Attrition_Flag", )  # https://seaborn.pydata.org/introduction.html https://seaborn.pydata.org/generated/seaborn.catplot.html
    plt.show()

def CHART_genderVS_platinum_card_program():
    df = import_data()
    df['Gender'].replace({'M': int(0), 'F': int(1)}, inplace=True)
    df['Attrition_Flag'].replace({'Existing Customer': int(0), 'Attrited Customer': int(1)},inplace=True)  # https://www.kite.com/python/answers/how-to-replace-column-values-in-a-pandas-dataframe-in-python
    df['Card_Category'].replace({'Blue': int(1), 'Silver': int(2), 'Gold': int(3), 'Platinum': int(4)})
    sns.barplot(x='Gender', y=('Card_Category'), hue='Attrition_Flag', data=df,ci=None)
    # Male is 0 and female is 1 so there are actually a lot of females leaving from the platinum card program.
    plt.show()

def correlation_analysis():
    df = import_data()
    df['Attrition_Flag'].replace({'Existing Customer': int(0), 'Attrited Customer': int(1)}, inplace=True)  # https://www.kite.com/python/answers/how-to-replace-column-values-in-a-pandas-dataframe-in-python
    df['Card_Category'].replace({'Blue': int(1), 'Silver': int(2), 'Gold': int(3), 'Platinum': int(4)})
    df['Gender'].replace({'M': int(0), 'F': int(1)}, inplace=True)

    df = df[['Attrition_Flag', 'Gender', 'Card_Category', 'Months_Inactive_12_mon', 'Contacts_Count_12_mon', 'Total_Relationship_Count', 'Total_Revolving_Bal', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio']]

    plt.figure(figsize=(10, 10))
    sns.heatmap(df.corr(),cbar=True, annot=True, cmap='YlGnBu') # https://seaborn.pydata.org/generated/seaborn.heatmap.html
    plt.ylim(13,0)

    plt.show()

def KNN(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state = 42)
    # Instantiate a k-NN classifier: knn
    knn = KNeighborsClassifier()

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    # Predict the labels of the test data: y_pred
    y_pred = knn.predict(X_test)

    # Generate the confusion matrix and classification report
    print('Untuned KNN Confusion Matrix: \n', confusion_matrix(y_test, y_pred))
    print('Untuned KNN Classification Report: \n', classification_report(y_test, y_pred))

def hyperparametertuning_onKNN_withRidge(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)
    # Setup the parameters and distributions to sample from: param_dist
    param_dist = {"n_neighbors": [5,6,7,8], "weights": ["uniform","distance"],"algorithm": ["auto", "ball_tree", "kd_tree","brute"], "leaf_size": [30,15,45]}

    # Instantiate KNN
    knn = KNeighborsClassifier()
    # Instantiate the RandomizedSearchCV object:
    knn_cv = RandomizedSearchCV(knn, param_dist, cv=5)

    # Fit it to the data -- Using the training data only as per: https://towardsdatascience.com/building-a-k-nearest-neighbors-k-nn-model-with-scikit-learn-51209555453a
    knn_cv.fit(X_train, y_train)

    # Print the tuned parameters and score
    print("Tuned Decision KNN Parameters: \n{}".format(knn_cv.best_params_))
    print("Tuned KNN best accuracy score is \n{}".format(knn_cv.best_score_))

    # Instantiate a ridge regressor: ridge
    ridge = Ridge(alpha=0.5)

    # Perform 5-fold cross-validation: ridge_cv -- using actual X, y as per: https://scikit-learn.org/stable/modules/cross_validation.html
    ridge_cv = cross_val_score(ridge, X, y, cv=5)

    # Print the cross-validated scores
    print('The "KNN-HyperparameterTuned - Ridge Cross Validation" scores: \n', ridge_cv) # https://towardsdatascience.com/the-power-of-ridge-regression-4281852a64d6
    # https://stats.stackexchange.com/questions/154706/how-to-interpret-ridge-regression-plot

def logreg_AUC(X, y): #https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)

    # Create the classifier: logreg
    logreg = LogisticRegression()

    # Fit the classifier to the training data
    logreg.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    # Compute predicted probabilities: y_pred_prob
    y_pred = logreg.predict(X_test)
    y_pred_prob = logreg.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) # https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Logistic Regression')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()

    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = logreg.predict_proba(X_test)[:, 1]

    # Compute and print AUC score
    print("Logistic Regression AUC: \n{}".format(roc_auc_score(y_test, y_pred_prob)))

    # Compute cross-validated AUC scores: cv_auc
    cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc') # https://www.mygreatlearning.com/blog/cross-validation/#:~:text=The%20purpose%20of%20cross%E2%80%93validation,generalize%20to%20an%20independent%20dataset.

    # Print list of AUC scores
    print("Logistic Regression - AUC scores computed using 5-fold cross-validation: \n{}".format(cv_auc))
    print("The mean of the 5 folds of AUC scores: ", cv_auc.mean())


def SVMwith_gridsearchHyperparametertuning(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)
    # Setup the pipeline
    steps = [('scaler', StandardScaler()), # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
             ('SVM', SVC())] # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    pipeline = Pipeline(steps) # https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    # Specify the hyperparameter space
    parameters = {'SVM__C': [1, 10, 100],
                  'SVM__gamma': [0.1, 0.01]}

    # Instantiate the GridSearchCV object: cv
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)

    # Fit to the training set
    cv.fit(X_train, y_train)

    # Predict the labels of the test set: y_pred
    y_pred = cv.predict(X_test)

    # Compute and print metrics
    print("SVM -  Pipeline Accuracy: \n{}".format(cv.score(X_test, y_test)))
    print(classification_report(y_test, y_pred))
    print("SVM  - Pipeline Tuned Model Parameters: \n{}".format(cv.best_params_))

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth = 1, random_state = 0).fit(X_train, y_train)
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    print('SVM after Gradient boosting - The CLF score is: \n', clf.score(X_test, y_test))


def main(): #the main function I nested the model functions in.
    df = import_data()
    df['Attrition_Flag'].replace({'Existing Customer': int(0), 'Attrited Customer': int(1)}, inplace=True)
    df['Card_Category'].replace({'Blue': int(1), 'Silver': int(2), 'Gold': int(3), 'Platinum': int(4)})
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    Customer_Attrition = df['Attrition_Flag']
    Features = pd.get_dummies(df[['Gender', 'Card_Category','Customer_Age', 'Dependent_count', 'Education_Level', 'Marital_Status',  'Income_Category', 'Months_on_book', 'Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Avg_Open_To_Buy','Total_Amt_Chng_Q4_Q1','Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1','Avg_Utilization_Ratio']])

    df = pd.concat([df, Customer_Attrition], axis=1)
    df = pd.concat([df, Features], axis=1)
    #df = df.drop(columns="CLIENTNUM")
    scaler = MinMaxScaler()

    Scaled_Features = scaler.fit_transform(Features)
    X = Scaled_Features
    y = Customer_Attrition
    oversample_minority(X,y) # oversampling the minority to try and get better model results.
    KNN(X,y) # Model type 1: K-Nearest Neighbors.
    hyperparametertuning_onKNN_withRidge(X,y) # Model type 1K-Nearest Neighbors, but now with hyperparameter tuning and ridge regression.
    logreg_AUC(X, y) # Model type 2: Logistic Regression with 'Area under the ROC curve analysis'.
    SVMwith_gridsearchHyperparametertuning(X,y) # Model type 3: Support-Vector Machines with parameter tuning and gradient boosting.

def oversample_minority(X, y):
    oversample = RandomOverSampler(sampling_strategy=0.5)
    # fit and apply the transform
    X, y = oversample.fit_resample(X, y)
    print('After oversampling minority - sampling counts [0=Existing_Customer and 1=Attrited_Customer]: \n', Counter(y))
    return X, y

main() # the main function I nested the model functions in.
#Leaving the below function calls to make it easier to test the separate functions:
#correlation_analysis()
#CHART_genderVS_platinum_card_program()
#CHART_Months_on_BooksVS_attrition()
#CHART_ContactsCountVS_attrition()
#CHART_TansactionsCountVS_attrition()
#exploring_data()
