import pandas
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import time

# Title
print("-----------------------------------------------------------------------------------------")
print("                               AUTOMATING DATA ANALYSIS                                   ")
print("-----------------------------------------------------------------------------------------")


# File Upload Management-------------------------------------------------------------------------------------
file_uploaded = False
while file_uploaded == False:
    file_path = input("Enter Data File Name: ")
    if ".xlsx" in file_path:
        try:
            file = pandas.read_excel(file_path)
            file_uploaded = True
        except:
            print("File Does Not Exist, Try Again!")
    elif ".csv" in file_path:
        try:
            file = pandas.read_csv(file_path)
            file_uploaded = True
        except:
            print("File Does Not Exist, Try Again!")
    else:
        print("File Does Not Exist, Try Again!")


#Functions For creating graphs, summarry reports, etc-----------------------------------------------------------
cols = file.columns
def create_graph(x, y, graph):
    if graph == "1":
        plt.scatter(file[x], file[y])
        plt.title("Scatter Plot")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
    elif graph == "2":
        plt.plot(file[x], file[y])
        plt.title("Lined Graph")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
    elif graph == "3":
        plt.bar(file[x], file[y])
        plt.title("Bar Graph")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()
    else:
        print("Error, Try Again!")

def summarry_report_col(file, col):
    print("----------------------------------------------------------------")
    print("--------------------" + col + " SUMMARRY REPORT------------------------")
    print("----------------------------------------------------------------\n")
    print(file[col].value_counts())


def summarry_report_full(file):
    print("----------------------------------------------------------------")
    print("--------------------FULL SUMMARRY REPORT------------------------")
    print("----------------------------------------------------------------\n")
    for x in range(len(cols)):  # iterates through that array
        print("-------------------------------------------------------------")
        print(cols[x] + " Report: ")
        print(file[cols[x]].value_counts())  # counts values of unique items
        print("-------------------------------------------------------------\n")

def print_cols(file):
    cols = file.columns
    print("Enter the number next to column to select")
    for x in range(len(cols)):
        print("(" + str(x) + ")" +  " " + cols[x])


def is_string(file, col):
    val = str(file[col].iat[0])
    if "1" in val or "2" in val or "3" in val or "4" in val or "5" in val or "6" in val or "7" in val or "8" in val or "9" in val or "0" in val:
        return False
    else:
        return True


# Main Loop-----------------------------------------------------------------------------------------------
print("-----------------------------------------------")
print("-----------------FILE UPLOADED-----------------")
print("-----------------------------------------------")
while True:
    print("\n---------------------------------------------------")
    print("Pick An Option, Then Enter The number next to it to start")
    print("(1) Summarry Report - Gives Summary Report On Data.")
    print("(2) Visualize Data - Visualizes data in any graph you pick")
    print("(3) Predict Future Data - Predicts certain value of Data")
    print("(4) QUIT")
    print("-----------------------------------------------------\n")

    choice = input("Enter Your Choice: ")
    if choice == "1":
        print("Do You Want To Summarize, A Specific Value or The Entire Dataframe?")
        print("(1) Specific Value")
        print("(2) Full Report")
        print("(3) Exit")
        summ = input("Enter The Number Next To Your Choice: ")
        if summ == "1":
            print_cols(file)
            column = int(input("What DO You Want To Summarize?: "))
            summarry_report_col(file, str(cols[column]))
        elif summ == "2":
            summarry_report_full(file)
        elif summ == "3":
            print("Leaving..")
        else:
            print("Try Again!")

    elif choice == "2":
        print("--------------------VISUALIZE DATA------------------------")
        print_cols(file)
        print("\n")
        x = int(input("Enter Your x Value or enter 9999 to QUIT: "))
        if x == "9999":
            print("Leaving..")
        else:
            y = int(input("Enter Your y Value: "))
            print("Graph Choices:")
            print("(1) ScatterPlot")
            print("(2) Lined Graph")
            print("(3) Bar Graph")
            graph = input("Enter Your Graph Choice: ")
            create_graph(cols[x], cols[y], graph)

    elif choice == "3":
        print("---------------------MACHINE LEARNING----------------------")
        # Input Value The User Wants to predict;
        # Input based on what the user wants to predict the value;
        # Use Every Machine Learning algorithm, do training and testing and display test results to user
        # Then Allow user to input few value to predict the PREDICTVALUE
        # Algorithms Used Include : Linear Regression, Support Vector Machines, K Nearest Neighbors etc;
        print_cols(file)
        predict = int(input("What do you want to predict? or enter 9999 to QUIT: "))
        if predict == "9999":
            print("Leaving..")
        else:
            # All ALgorithm models
            linear = linear_model.LinearRegression()
            knn = KNeighborsClassifier(n_neighbors=11)
            SVM = svm.SVC()
            tr = tree.DecisionTreeClassifier()
            le = preprocessing.LabelEncoder()

            print_cols(file)
            based = int(input("Based On what would you like to predict " + str(cols[predict]) + ": "))
            # based = cols[based]
            data = file[[cols[based], cols[predict]]]

            # based and predict are nums
            # convert them to their string
            # check if they are string
            # if yes, labelencode
            # else continue
            # ------------------------------------------------------------------------------------------------------

            # y is want you want to predict
            # x is what you are using to predict y
            # x true y false
            # x false y true
            # x true y true
            # x false y false
            x_string = False
            y_string = False

            if is_string(file, cols[based]) == True and is_string(file, cols[predict]) == False:
                based_option = le.fit_transform(list(data[cols[based]]))
                X = np.array([based_option])
                y = np.array(data[[cols[predict]]])
                X = X.transpose()
                x_string = True
                #WORKS

            elif is_string(file, cols[based]) == False and is_string(file, cols[predict]) == True:
                predict_option = le.fit_transform(list(data[cols[predict]]))
                X = np.array(data.drop([cols[predict]], 1))
                y = np.array(predict_option)
                y_string = True
                #WORKS

            elif is_string(file, cols[based]) == True and is_string(file, cols[predict]) == True:
                based_option = le.fit_transform(list(data[cols[based]]))
                predict_option = le.fit_transform(list(data[cols[predict]]))
                X = np.array([based_option])
                y = np.array(predict_option)
                X = X.transpose()
                x_string = True
                y_string = True

            else:
                X = np.array(data.drop([cols[predict]], 1))
                y = np.array(data[[cols[predict]]])

            # ----------------------------------------------------------------------------------------------------------

            # Creating Training Data
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3)


            # Determining which algorithm is best
            linear.fit(x_train, y_train)
            lacc = linear.score(x_test, y_test)

            knn.fit(x_train, y_train)
            kacc = knn.score(x_test, y_test)

            SVM.fit(x_train, y_train)
            sacc = SVM.score(x_test, y_test)

            tr.fit(x_train, y_train)
            tacc = tr.score(x_test, y_test)

            print("AI 1 Efficiency: " + str(round(lacc * 100)))
            print("AI 2 Efficiency: " + str(round(kacc * 100)))
            print("AI 3 Efficiency: " + str(round(sacc * 100)))
            print("AI 4 Efficiency: " + str(round(tacc * 100)))

            #Determining Best ALgorithm
            list = [lacc, kacc, sacc, tacc]
            select = max(list)


            # Issues To adress or fix:
            # 1) The predict part input.
            # 2) Add many more algorithms for better chance and proficiency
            # 3) Convert to exe
            # 4) Create logo and give a name to software

            # Linear Regression-----------------------------------------------------------------------------------
            if select == lacc:
                time.sleep(3)
                print("--------------------------AI Selected--------------------------")
                time.sleep(2)
                print("AI 1 Efficiency: " + str(round(lacc * 100)))
                time.sleep(3)
                print("Test Results: ")
                time.sleep(2)
                predictions = linear.predict(x_test)
                if x_string == True:
                    x_test = le.inverse_transform(x_test)

                if y_string == True:
                    print(le.classes_)
                    predictions = le.inverse_transform(predictions)
                    y_test = le.inverse_transform(y_test)
                for x in range(len(predictions)):
                    # predictions[x]  what computer predicted
                    # x_test[x]       What is being used to predict
                    # y_test[x]       what is being predicted
                    print("-------------------------------------")
                    print(cols[based] + ": " + str(x_test[x]))
                    print("AI Prediction For " + cols[predict] + ": " + str(predictions[x]))
                    print("Actual " + cols[predict] + ": " + str(y_test[x]))
                print("\n\n")
                predict_option = int(input("Enter a " + cols[based] + ": "))

                array = [[predict_option]]
                predictions = linear.predict(array)

                for x in range(len(predictions)):
                    print("Predicted " + cols[predict] + ": " + str(predictions[x]))
                x_string = False
                y_string = False


            # K Nearest Neighbors----------------------------------------------------------------------------------
            elif select == kacc:
                time.sleep(3)
                print("--------------------------AI Selected--------------------------")
                time.sleep(2)
                print("AI 2 Efficiency: " + str(round(kacc * 100)))
                time.sleep(3)
                print("Test Results: ")
                time.sleep(2)
                predictions = knn.predict(x_test)
                if x_string == True:
                    x_test = le.inverse_transform(x_test)

                if y_string == True:
                    print(le.classes_)
                    predictions = le.inverse_transform(predictions)
                    y_test = le.inverse_transform(y_test)
                for x in range(len(predictions)):
                    # predictions[x]  what computer predicted
                    # x_test[x]       What is being used to predict
                    # y_test[x]       what is being predicted
                    print("-------------------------------------")
                    print(cols[based] + ": " + str(x_test[x]))
                    print("AI Prediction For " + cols[predict] + ": " + str(predictions[x]))
                    print("Actual " + cols[predict] + ": " + str(y_test[x]))
                print("\n\n")
                predict_option = int(input("Enter a " + cols[based] + ": "))

                array = [[predict_option]]
                predictions = knn.predict(array)

                for x in range(len(predictions)):
                    print("Predicted " + cols[predict] + ": " + str(predictions[x]))
                x_string = False
                y_string = False

            #Support Vector Machines-------------------------------------------------------------------------------
            elif select == sacc:
                time.sleep(3)
                print("--------------------------AI Selected--------------------------")
                time.sleep(2)
                print("AI 3 Efficiency: " + str(round(sacc * 100)))
                time.sleep(3)
                print("Test Results: ")
                time.sleep(2)
                predictions = SVM.predict(x_test)
                if x_string == True:
                    x_test = le.inverse_transform(x_test)

                if y_string == True:
                    print(le.classes_)
                    predictions = le.inverse_transform(predictions)
                    y_test = le.inverse_transform(y_test)
                for x in range(len(predictions)):
                    # predictions[x]  what computer predicted
                    # x_test[x]       What is being used to predict
                    # y_test[x]       what is being predicted
                    print("-------------------------------------")
                    print(cols[based] + ": " + str(x_test[x]))
                    print("AI Prediction For " + cols[predict] + ": " + str(predictions[x]))
                    print("Actual " + cols[predict] + ": " + str(y_test[x]))
                print("\n\n")
                predict_option = int(input("Enter a " + cols[based] + ": "))

                array = [[predict_option]]
                predictions = SVM.predict(array)

                for x in range(len(predictions)):
                    print("Predicted " + cols[predict] + ": " + str(predictions[x]))
                x_string = False
                y_string = False


            #Decision Trees--------------------------------------------------------------------------------------
            elif select == tacc:
                time.sleep(3)
                print("--------------------------AI Selected--------------------------")
                time.sleep(2)
                print("AI 4 Efficiency: " + str(round(tacc * 100)))
                time.sleep(3)
                print("Test Results: ")
                time.sleep(2)
                predictions = tr.predict(x_test)
                if x_string == True:
                    x_test = le.inverse_transform(x_test)

                if y_string == True:
                    print(le.classes_)
                    predictions = le.inverse_transform(predictions)
                    y_test = le.inverse_transform(y_test)
                for x in range(len(predictions)):
                    # predictions[x]  what computer predicted
                    # x_test[x]       What is being used to predict
                    # y_test[x]       what is being predicted
                    print("-------------------------------------")
                    print(cols[based] + ": " + str(x_test[x]))
                    print("AI Prediction For " + cols[predict] + ": " + str(predictions[x]))
                    print("Actual " + cols[predict] + ": " + str(y_test[x]))
                print("\n\n")
                predict_option = int(input("Enter a " + cols[based] + ": "))

                array = [[predict_option]]
                predictions = tr.predict(array)

                for x in range(len(predictions)):
                    print("Predicted " + cols[predict] + ": " + str(predictions[x]))
                x_string = False
                y_string = False
    elif choice == "4":
        exit()

    else:
        print("Error, Try Again!")