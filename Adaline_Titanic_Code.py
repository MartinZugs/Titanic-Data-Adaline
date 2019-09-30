import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from Adaline_SGD import AdalineSGD

def clean_up(file):
    #This logic fills in the age with the median of the other properly populated data
    file["Age"] = file["Age"].fillna(file["Age"].dropna().median())
    

    #C - Cherbourg 0, S - Southampton 1, Q = Queenstown 2
    file["Embarked"] = file["Embarked"].fillna("S")
    file.loc[file["Embarked"] == "C", "Embarked"] = 0
    file.loc[file["Embarked"] == "S", "Embarked"] = 1
    file.loc[file["Embarked"] == "Q", "Embarked"] = 2

    #Male = 0, Female = 1
    file.loc[file["Sex"] == "male", "Sex"] = 0
    file.loc[file["Sex"] == "female", "Sex"] = 1

train = pd.read_csv("./train.csv")
keep_col = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
new_f = train[keep_col]
new_f.to_csv("train_clean.csv", index=False)

train = pd.read_csv("./train_clean.csv")

#This deals with issues in the data being unusable for ML algorithms because it is non-numeric
clean_up(train)

# Divide the data into almost exactly 70% train and 30% test
y = train.iloc[0:623, 0].values
X = train.iloc[0:623, [1, 2, 3, 4, 5]].values
y_test = train.iloc[624:, 0].values
x_test = train.iloc[624:, [1, 2, 3, 4, 5]].values

# Create the AdalineSGD model
model2 = AdalineSGD(n_iter = 100, eta = 0.0001)

# Train the model
model2.fit(X, y)

# Plot the training error
plt.plot(range(1, len(model2.cost_) + 1), model2.cost_, marker = 'x', color = 'blue')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()


# Predict for the test set
t = 0
correct = 0
false = 0

for p in x_test:
    #print(p)
    predict = model2.predict(p)
    #print(predict)
    #print(y_test[t])

    if (predict == -1):
        predict = 0

    if (predict == y_test[t]):
        #print("Correct!")
        false = false + 1
    else:
        #print("False")
        correct = correct + 1
        
    t = t+1

# Print the percent accuracy
print("Overall: " + str(correct/len(x_test)))
