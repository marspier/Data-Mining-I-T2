# Data-Mining-I-T2




Part I: Research Question

A. 

1.  My question for this analysis is, can I predict the type of internet service (DSL, fiber optic, or None) that a new customer will  likely choose based on certain factors? 

•  decision trees

2.  My goal for the data analysis is to develop a predictive model that accurately predicts if certain variables can predict the type of internet service (DSL, fiber optic, or None) a customer will choose.
I want to help a business implement tools to market to the right audience when it comes to the different types of internet services. 

Part II: Method Justification

B.  
1.  Explain how the prediction method you chose analyzes the selected data set. Include expected outcomes.

The Decision Tree method analyzes the dataset by evaluating different  factors, such as demographics and preferences, to create a tree-like structure that organizes  customers into smaller groups. By doing this, we can then predict a customer service choice. According to Chauhan, “The goal of using a Decision Tree is to create a training model that can be used to predict the class or value of the target variable by learning simple decision rules inferred from prior data(training data).” 


2.  Summarize one assumption of the chosen prediction method.

One assumption is that decision trees are non-parametric. According to IBM, decision trees operate as a non-parametric algorithm, suitable for both classification and regression, and structured with a hierarchy of nodes and branches (IBM, n.d.). This means that decision trees make decisions based entirely on the data provided, rather than on prior assumptions.

3.  List the packages or libraries you have chosen for Python or R, and justify how each item on the list supports the analysis.

import numpy as np:Numerical computations

import pandas as pd:Provides dataframe 

import seaborn as sns:Create plots

from sklearn.model_selection import train_test_split: Split the data 

from sklearn.tree import DecisionTreeClassifier: To build  and train the decision tree model 

from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,precision_score,recall_score,f1_score: To evaluate the performance of the classification model

import matplotlib.pyplot as plt %matplotlib inline: Create plots
from sklearn.feature_selection import SelectKBest: feature selection method that selects the top k features based on a specified score function
from sklearn.feature_selection import f_classif:f_classif is a score function used for classification tasks
from sklearn.model_selection import GridSearchCV: hyperparameter tuning using cross-validation
from sklearn.metrics import mean_squared_error: To get the average squared difference between the predicted values and the actual values
from sklearn.tree import plot_tree: visual representation of the decision tree model
from sklearn.model_selection import cross_val_score: for performing cross-validation to evaluate the model's performance
from sklearn.model_selection import learning_curve: helps visual overfitting or underfitting of the models performance

Part III: Data Preparation

C.  Perform data preparation for the chosen data set by doing the following:

1.  Describe one data preprocessing goal relevant to the prediction method from part A1.

One data preprocessing goal relevant to decision trees is handling categorical variables, such as the categories of internet services in this case. Decision trees are flexible and can manage both discrete and continuous data types, which makes them suitable for a variety of applications. According to IBM, "decision trees have a number of characteristics, which make  it more flexible than other classifiers. It can handle various data types—i.e., discrete or continuous values, and continuous values can be converted into categorical values through the use of thresholds" (IBM, n.d.).
Having this in mind, I would say that  a preprocessing goal would be to encode categorical variables into numerical format, such as using Get_dummies encoding, so that the decision tree algorithm can effectively utilize these variables during the splitting process. This preprocessing step helps ensure that the decision tree accurately captures relationships between categorical variables and the target variable during analysis.


2.  Identify the initial data set variables that you will use to perform the analysis for the prediction question from part A1, and group each variable as continuous or categorical. 


| Categorical Variables       | Continuous Variables      |
|-----------------------------|--------------------------|
| Email                       | Age                      |
| Yearly_equip_failure        | Income                   |
| DummyGender                 | Outage_sec_perweek       |
| DummyChurn                  | Tenure                   |
| DummyMarital                | MonthlyCharge            |
| DummyTechie                 | Bandwidth_GB_Year        |
| DummyContract               | Children                 |
| DummyPort_modem             |                          |
| DummyTablet                 |                          |
| DummyPhone                  |                          |
| DummyMultiple               |                          |
| DummyOnlineSecurity         |                          |
| DummyOnlineBackup           |                          |
| DummyDeviceProtection       |                          |
| DummyTechieSupport          |                          |
| DummyStreamingTV            |                          |
| DummyStreamingMovies        |                          |





1.I imported the libraries and packages 
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve




 Encoded my categorical variables into dummy variables to include them in analysis that are only compatible with numerical data. 
Encode categorical variables into dummy variables
df['DummyChurn'] = [1 if v == 'Yes' else 0 for v in df['Churn']]
df['DummyGender'] = [1 if v == 'Male' else 0 for v in df['Gender']]
df['DummyTechie'] = [1 if v == 'Yes' else 0 for v in df['Techie']]
df['DummyContract'] = [1 if v == 'Two Year' else 0 for v in df['Contract']]
df['DummyPort_modem'] = [1 if v == 'Yes' else 0 for v in df['Port_modem']]
df['DummyTablet'] = [1 if v == 'Yes' else 0 for v in df['Tablet']]
df['DummyPhone'] = [1 if v == 'Yes' else 0 for v in df['Phone']]
df['DummyMultiple'] = [1 if v == 'Yes' else 0 for v in df['Multiple']]
df['DummyOnlineSecurity'] = [1 if v == 'Yes' else 0 for v in df['OnlineSecurity']]
df['DummyOnlineBackup'] = [1 if v == 'Yes' else 0 for v in df['OnlineBackup']]
df['DummyDeviceProtection'] = [1 if v == 'Yes' else 0 for v in df['DeviceProtection']]
df['DummyTechSupport'] = [1 if v == 'Yes' else 0 for v in df['TechSupport']]
df['DummyStreamingTV'] = [1 if v == 'Yes' else 0 for v in df['StreamingTV']]
df['DummyStreamingMovies'] = [1 if v == 'Yes' else 0 for v in df['StreamingMovies']]
df['DummyMarital'] = df['Marital'].replace(['Divorced','Widowed','Separated','Never Married'],'NotMarried')
df['DummyMarital'] = [1 if v == 'Married' else 0 for v in df['Marital']]
df['TargetInternetService'] = df['InternetService'].replace({'None': 0, 'DSL': 1, 'Fiber Optic': 2})


13. I then dropped the non dummy variables 
df=df.drop(columns=['Churn','Gender','Marital','Techie','Contract','Port_modem','Tablet','Phone','InternetService','Multiple','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies'])




14. I  organized my dataset and added the dependent variable at the end. 
 Select final set of features
df = df[['Children', 'Age', 'Income', 'Outage_sec_perweek','Email',
'Yearly_equip_failure','Tenure','MonthlyCharge','Bandwidth_GB_Year','DummyGender','DummyChurn','DummyTechie','DummyContract', 'DummyPort_modem', 'DummyTablet', 
'DummyPhone','DummyMultiple', 'DummyOnlineSecurity', 
'DummyOnlineBackup','DummyDeviceProtection', 'DummyTechSupport', 'DummyStreamingTV','DummyStreamingMovies',
'DummyMarital','TargetInternetService']]



15. I created a correlation heatmap to have a visual of any correlations in the dataset. 



Create Heatmap


selected_variables = ['Children', 'Age','Income','Outage_sec_perweek','Email','Yearly_equip_failure',
'Tenure','MonthlyCharge','Bandwidth_GB_Year','DummyGender','DummyChurn','DummyMarital','DummyTechie','DummyContract', 'DummyPort_modem','DummyTablet','DummyPhone',
'DummyMultiple','DummyOnlineSecurity','DummyOnlineBackup','DummyDeviceProtection',
'DummyTechSupport','DummyStreamingTV','DummyStreamingMovies','TargetInternetService']
correlation_matrix = df[selected_variables].corr()
plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()





16. I separated the independent variables from the dependent variable. 

Separate X and Y values 
X = df[['Children', 'Age', 'Income', 'Outage_sec_perweek', 'Email', 'Yearly_equip_failure',
'Tenure', 'MonthlyCharge', 'Bandwidth_GB_Year', 'DummyGender', 'DummyChurn', 'DummyMarital', 'DummyTechie','DummyContract', 'DummyPort_modem', 'DummyTablet', 
'DummyPhone', 'DummyMultiple', 'DummyOnlineSecurity','DummyOnlineBackup', 
'DummyDeviceProtection', 'DummyTechSupport','DummyStreamingTV',
'DummyStreamingMovies']]
y = df['TargetInternetService']


17. I split the data into training and testing 
Train_Test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

18. I finally saved the training and testing sets as CSV files. 
save training and testing sets
X_train.to_csv('X_train1.csv', index=False)
X_test.to_csv('X_test1.csv', index=False)
y_train.to_csv('y_train1.csv', index=False)
y_test.to_csv('y_test1.csv', index=False)

1.  Split the data into training and test data sets and provide the file(s).
X_test X_train
Y_test Y_train

 
2.  Describe the analysis technique you used to appropriately analyze the data. Include screenshots of the intermediate calculations you performed.

I first prepared my data by checking for any missing values, duplicate values, and outliers. I also encoded categorical variables into numerical ones by using dummy variables.  Once I cleaned my data, I performed univariate and bivariate analyses. This allowed me to explore the data by involving visualizations such as histograms to see the relationship between variables. I was then ready to start the analysis for my decision tree model. 

I used the prepared dataset and I split the data into training and testing sets. I then used a decision tree model to predict the type of internet service that customers will choose.  After I had trained the decision tree classifier, I evaluated the initial model performance by printing the accuracy, precision, recall, and F-1 score. I then set up  hyperparameters by using Grid Search. Since I am working with a decision tree model, I do not want to risk an overfitting model that can lead to false predictions. To prevent this, I fine-tune the max depth and the minimum number of samples required to split an internal node. After tuning the model, I evaluated its performance on the test set and compared it with the initial results. I calculated metrics such as accuracy, precision, recall, and F1-score to assess the model's predictive capability.

I finished my analysis with visuals such as the learning curve for the decision tree, and the plot tree as well as the feature importance bar chart. 



1.  Explain the accuracy and the mean squared error (MSE) of your prediction model.



Initial 
Accuracy: 1.0
Precision: 1.0
Recall: 1.0
F1-score: 1.0
Confusion Matrix:
[[1725    0]
 [   0 6275]]
Mean Squared Error (Training set): 0.0
Root Mean Squared Error (Training set): 0.0

Testing set
Accuracy: 0.9155
Precision: 0.9151
Recall: 0.9155
F1-score: 0.9152
Confusion Matrix:
[[341  37  26]
 [ 32 632  33]
 [ 23  18 858]]

Test Mean Squared Error (MSE): 0.158
Test Root Mean Squared Error (RMSE): 0.39749213828703583


The Mean Squared Error (MSE) is the average of the squared differences between predicted values and actual values in the dataset. Therefore, a low MSE indicates a small difference between the predicted and actual values.
The Root Mean Squared Error (RMSE) measures the average difference between the predicted models and the actual values on the same scale as the original data. A lower RMSE value is preferred, as it indicates a smaller difference between the predicted values and the actual values.
In this case, the MSE value of 0.158 shows that there is some error, but it is not significant. The RMSE of 0.39 is also relatively low and provides a better perspective on the model's performance. I like to think of RMSE as a measure of how significant the error truly is. While the MSE of 0.158 may seem small, the RMSE of 0.39 puts it into context, demonstrating that the error, while small, is more meaningful in terms of the scale of the actual values.
This understanding helps us take these errors into account, enabling us to make well-informed decisions.


2.  Discuss the results and implications of your prediction analysis.

The initial accuracy, precision, recall and F1-score all have a 1.0 which is a sign there may be an overfitting of the model. 
Since the results were 1.0 it makes sense why the Mean for the squared error was 0.0 and the Root mean squared error is 0.0.
The results from the testing set indicate a well-performing model. The accuracy of 0.9155 shows  that approximately 91.55% of the predictions made by the model were correct which is a strong overall performance. The precision of 0.9151 reflects that when the model predicts a positive class, about 91.51% of those predictions are correct. Meanwhile, the recall of 0.9155 indicates that the model successfully identifies 91.55% of the actual positive cases. The F1-score, calculated at 0.9152, serves as a balanced measure that combines both precision and recall, further. 
In the matrix, the diagonal values (341, 632, and 858) represent the true positives for each class. We can see that there are  37 instances of class 0 misclassified as class 1, 26 instances of class 0 misclassified as class 2, and 32 instances of class 1 misclassified as class 0. Then we see 23 instances where class 2 was misclassified as class 0. 18 instances where class 2 was misclassified as class 1.  Although the model performs well and the low number of misclassifications compared to the correct numbers still suggests potential improvement. 

The mean CV score of .903 (90%)  indicates that on average my model achieved a high level of accuracy across different subsets of the data. 
The standard deviation for cross-validation, with a score of .007, suggests that the model's performance is consistent across folds since it’s a small standard deviation. 
Overall these results show that the model is effective in predicting customer choices regarding internet service options. 

3.  Discuss one limitation of your data analysis.

One limitation of decision trees is the potential for overfitting the trained data. As Pramod explains, “A decision tree will always overfit the training data if we allow it to grow to its maximum depth. Overfitting occurs when the tree becomes too complex and captures noise in the training data rather than the underlying pattern.”
This can be missed since overfitting data can result in a high accuracy result on the training data. However, if one relies solely on the high accuracy of the training data it can be misleading, since it may indicate overfitting and lead to inaccurate predictions. 

In my learning curve model, the training score is consistently higher than the cross-validation score. This pattern suggests that the model may be overfitting the training data. However, the gradual increase in the cross-validation score indicates that the model is improving as more data is added, which suggests it is learning from the data effectively.
As previously mentioned, one assumption of this model is that it does not rely on preconceived notions. Instead, it adapts to the data provided. The trend of increasing cross-validation scores demonstrates that as we increase the amount of training data, the model becomes better at capturing the underlying patterns, leading to improved performance on unseen examples.

4. Recommend a course of action for the real-world organizational situation from part A1 based on your results and implications discussed in part E2.


Aside from the fact that my model is performing well with a high accuracy percentage, I can show  significant insights from the decision tree plot. At the root node, we observe that when the monthly charge is less than or equal to $143.74, customers are further classified based on their bandwidth usage. Specifically, for customers with monthly charges less than or equal to $107, the service tends to be DSL. This suggests that customers who choose lower price services are often associated with DSL service. 
When the bandwidth is less than or equal to 732, it appears that customers still prefer DSL. However, as the bandwidth increases above this threshold, customers begin to shift towards higher-end services. If the bandwidth exceeds 6162, the classification shifts toward fiber optic services, indicating a strong preference for faster internet among customers willing to pay more.
Given these insights, I recommend promoting fiber optic services to customers with higher bandwidth needs. We should emphasize the benefits of fiber optic service, particularly the superior speed it offers. Additionally, we could consider offering discounts or promotional packages to incentivize upgrades to fiber optic services.
In summary,  my findings indicate that customers who pay higher monthly charges tend to opt for fiber optic services, which also correlate with higher bandwidth usage. This suggests that these customers may engage in more high bandwidth  activities, such as gaming or streaming.
I would additionally  recommend further advertising targeted at this demographic. I would also like to add that gathering more data on customer behavior could help refine our marketing strategies and identify potential new customer segments who would benefit from our offerings.


Bobbitt, Z. (2021, September 30). MSE vs. RMSE: What's the difference? Statology. https://www.statology.org/mse-vs-rmse/

Brownlee, Jason. “Overfitting in Machine Learning Models.” Machine Learning Mastery, 27 Nov. 2020, machinelearningmastery.com/overfitting-machine-learning-models/.

Frost, Jim. "Root Mean Square Error (RMSE)." Statistics by Jim, statisticsbyjim.com/regression/root-mean-square-error-rmse/.

Chauhan, Nagesh Singh. "Decision Tree Algorithm Explained." KDnuggets, 9 Feb. 2022, www.kdnuggets.com/2020/01/decision-tree-algorithm-explained.html.

IBM. (n.d.). What is a decision tree? Retrieved from https://www.ibm.com/cloud/learn/decision-trees 

Pramod, Om. "Decision Trees." Medium, 29 Jan. 2023, medium.com/@ompramod9921/decision-trees-8e2391f93fa7.


