import pandas
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn import svm
from sklearn import metrics
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
    cols = file.columns  # Array of every column value in dataframe
    for x in range(len(cols)):  # iterates through that array
        print("-------------------------------------------------------------")
        print(cols[x] + " Report: ")
        print(file[cols[x]].value_counts())  # counts values of unique items
        print("-------------------------------------------------------------\n")

def print_cols(file):
    cols = file.columns
    for x in range(len(cols)):
        print("(" + str(x) + ")" +  " " + cols[x])


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
            column = input("What DO You Want To Summarize?: ")
            summarry_report_col(file, column)
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
        x = input("Enter Your x Value or enter 9999 to QUIT: ")
        if x == "9999":
            print("Leaving..")
        else:
            y = input("Enter Your y Value: ")
            print("Graph Choices:")
            print("(1) ScatterPlot")
            print("(2) Lined Graph")
            print("(3) Bar Graph")
            graph = input("Enter Your Graph Choice: ")
            create_graph(x, y, graph)

    elif choice == "3":
        print("---------------------MACHINE LEARNING----------------------")
        # Input Value The User Wants to predict;
        # Input based on what the user wants to predict the value;
        # Use Every Machine Learning algorithm, do training and testing and display test results to user
        # Then Allow user to input few value to predict the PREDICTVALUE
        # Algorithms Used Include : Linear Regression, Support Vector Machines, K Nearest Neighbors etc;
        print_cols(file)
        predict = input("What do you want to predict? or enter 9999 to QUIT: ")
        if predict == "9999":
            print("Leaving..")
        else:
            print("DO you want to predict " + predict + " based on multiple factors or one factor?")
            print("(1) Multiple Values")
            print("(2) Single Value")
            multiple = input("Enter the number next to your choice: ")
            if multiple == "1":
                # Multiple Values
                print("Not yet coded....")
            elif multiple == "2":
                print_cols(file)
                based = input("Based On what would you like to predict " + predict + ": ")
                data = file[[based, predict]]
                try:
                    X = np.array(data.drop([predict], 1))
                    y = np.array(data[predict])
                except:
                    print("Error, Values Not Numerical")

                # Creating Training Data
                x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.01)

                # All ALgorithm models
                linear = linear_model.LinearRegression()
                knn = KNeighborsClassifier(n_neighbors=1)
                SVM = svm.SVC(kernel="linear")

                # Converting String TO predictable form

                # Determining which algorithm is best
                linear.fit(x_train, y_train)
                lacc = linear.score(x_test, y_test)
                knn.fit(x_train, y_train)
                kacc = knn.score(x_test, y_test)
                SVM.fit(x_train, y_train)
                sacc = SVM.score(x_test, y_test)
                print("Algorithm 1 Efficiency: " + str(round(lacc*100)))
                print("Algorithm 2 Efficiency: " + str(round(kacc*100)))
                print("Algorithm 3 Efficiency: " + str((round(sacc))))
                if lacc > kacc:
                    time.sleep(3)
                    print("--------------------------Algorithm Selected--------------------------")
                    time.sleep(2)
                    print("Algorithm Efficiency: " + str(round(lacc * 100)))
                    time.sleep(3)
                    print("Test Results: ")
                    time.sleep(2)
                    predictions = linear.predict(x_test)
                    for x in range(len(predictions)):
                        # predictions[x]  what computer predicted
                        # x_test[x]       What is being used to predict
                        # y_test[x]       what is being predicted
                        print("-------------------------------------")
                        print("Algorithm Prediction For " + predict + ": " + str(predictions[x]))
                        print("Actual " + predict + ": " + str(y_test[x]))
                    print("\n\n")
                    predict_option = int(input("Enter a " + based + ": "))
                    array = [[predict_option]]
                    predictions = linear.predict(array)
                    for x in range(len(predictions)):
                        print("Predicted " + predict + ": " + str(predictions[x]))
                elif kacc > lacc:
                    time.sleep(3)
                    print("--------------------------Algorithm Selected--------------------------")
                    time.sleep(2)
                    print("Algorithm Efficiency: " + str(round(kacc * 100)))
                    time.sleep(3)
                    print("Test Results: ")
                    time.sleep(2)
                    predictions = knn.predict(x_test)
                    for x in range(len(predictions)):
                        # predictions[x]  what computer predicted
                        # x_test[x]       What is being used to predict
                        # y_test[x]       what is being predicted
                        print("-------------------------------------")
                        print("Algorithm Prediction For " + predict + ": " + str(predictions[x]))
                        print("Actual " + predict + ": " + str(y_test[x]))
                    print("\n\n")
                    predict_option = int(input("Enter a " + based + ": "))
                    array = [[predict_option]]
                    predictions = knn.predict(array)
                    for x in range(len(predictions)):
                        print("Predicted " + predict + ": " + str(predictions[x]))
                elif sacc > lacc:
                    time.sleep(3)
                    print("--------------------------Algorithm Selected--------------------------")
                    time.sleep(2)
                    print("Algorithm Efficiency: " + str(round(kacc * 100)))
                    time.sleep(3)
                    print("Test Results: ")
                    time.sleep(2)
                    predictions = SVM.predict(x_test)
                    for x in range(len(predictions)):
                        # predictions[x]  what computer predicted
                        # x_test[x]       What is being used to predict
                        # y_test[x]       what is being predicted
                        print("-------------------------------------")
                        print("Algorithm Prediction For " + predict + ": " + str(predictions[x]))
                        print("Actual " + predict + ": " + str(y_test[x]))
                    print("\n\n")
                    predict_option = int(input("Enter a " + based + ": "))
                    array = [[predict_option]]
                    predictions = knn.predict(array)
                    for x in range(len(predictions)):
                        print("Predicted " + predict + ": " + str(predictions[x]))
                elif sacc > kacc:
                    time.sleep(3)
                    print("--------------------------Algorithm Selected--------------------------")
                    time.sleep(2)
                    print("Algorithm Efficiency: " + str(round(kacc * 100)))
                    time.sleep(3)
                    print("Test Results: ")
                    time.sleep(2)
                    predictions = knn.predict(x_test)
                    for x in range(len(predictions)):
                        # predictions[x]  what computer predicted
                        # x_test[x]       What is being used to predict
                        # y_test[x]       what is being predicted
                        print("-------------------------------------")
                        print("Algorithm Prediction For " + predict + ": " + str(predictions[x]))
                        print("Actual " + predict + ": " + str(y_test[x]))
                    print("\n\n")
                    predict_option = int(input("Enter a " + based + ": "))
                    array = [[predict_option]]
                    predictions = SVM.predict(array)
                    for x in range(len(predictions)):
                        print("Predicted " + predict + ": " + str(predictions[x]))

    elif choice == "4":
        exit()

    else:
        print("Error, Try Again!")
