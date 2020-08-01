# Linear Regression
In simple words, Linear regression is a simple model with a best fit line called the regression line
### Equation
#### Simple Linear Regression
  1. y = mx + c, where m is the slope and c the y intercepts
  2. m = (y1 - y2) / (x1 - x2) , where x and y are points in  the line
#### Multiple Linear Regression
   y=M1X1+M2X2+...+C
## DISADVANTAGE
  It will work if only a linear relationship exists between the Dependent and the Independent variables
# Project
In this project we will be implementing the linear regression algorithm to predict students final grade based on a series of attributes
  1. [Student](./student-mat.csv "Dataset")
        The dataset consists of student details along with their grades
        
        Understanding the dataset is a vital part in machine learning, so go through the dataset
        
        we would use only "G1", "G2", "G3", "studytime", "failures", "absences" these attributes to build our model
  2. [Linear_Regression](./linear-regression.py "Code")
         This above link consists of the entire code 
  3. The trained model with the best accuracy is stored in [Linear_Regression_Model](./studentgrades.pickle "Model")
## Working
  1. Import Modules
      ```
         import pandas as pd
         import numpy as np
         import sklearn
         from sklearn import linear_model
         from sklearn.utils import shuffle
      ```   
  2. Loading in Our Data
      ```
         data = pd.read_csv("student-mat.csv", sep=";")
         # Since our data is seperated by semicolons we need to do sep=";"
      ```   
  3. Trimming Our Data
      ```
      data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
      ```
  4. Separating Our Data
    ```
      predict = "G3"
      X = np.array(data.drop([predict], 1)) # Features
      Y = np.array(data[predict]) # Labels
      x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)
    ```
      The above code would split the data into 4 sets
 5. Implementating the Algorithm
    ```
      linear = linear_model.LinearRegression()  # create model
      linear.fit(x_train, y_train)              #Train model
      accuracy = linear.score(x_test, y_test)   #Check accuracy
     ```
 6. Viewing The Constants
    ```
      print('Coefficient: \n', linear.coef_) # These are each slope value
      print('Intercept: \n', linear.intercept_) # This is the intercept
  
 7. Predicting Students Grades
    ```
      predictions = linear.predict(x_test) # Gets a list of all predictions
      for x in range(len(predictions)):
          print(predictions[x], x_test[x], y_test[x])
      
    ```
 8. Saving Our Model
    ```
        import matplotlib.pyplot as plt
        from matplotlib import style
        import pickle
        with open("studentgrades.pickle", "wb") as f:
            pickle.dump(linear, f)

        # linear is the name of the model we created in the last tutorial
        # it should be defined above this
 
 9.  Loading Our Model
   ```
        pickle_in = open("studentgrades.pickle", "rb")
        linear = pickle.load(pickle_in)
        # Now we can use linear to predict grades like before
   ```
10. Training Multiple Models
      ```
        # TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
        best = 0
        for _ in range(20):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

            linear = linear_model.LinearRegression()

            linear.fit(x_train, y_train)
            acc = linear.score(x_test, y_test)
            print("Accuracy: " + str(acc))

            # If the current model has a better score than one we've already trained then save it
            if acc > best:
                best = acc
                with open("studentgrades.pickle", "wb") as f:
                    pickle.dump(linear, f)
        
 11.  Plotting Our Data
      ```
        # Drawing and plotting model
        plot = "failures" # Change this to G1, G2, studytime or absences to see other graphs
        plt.scatter(data[plot], data["G3"]) 
        plt.legend(loc=4)
        plt.xlabel(plot)
        plt.ylabel("Final Grade")
        plt.show()
      ```
