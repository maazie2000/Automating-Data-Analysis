# MLBot


MLBot is a software application that can be used to simplify the processes of data analysis and machine learning. The software can parse a dataset of your choice and 
create a summarray report, visualize data in any graph and use machine learning to predict future data.



# 1) Uploading A Dataset

The software can input a csv(comma-separated values) file or a xlsx(excel) file. To upload a file, the dataset and software should be in a local directory. After your 
dataset is in the same directory as the software, you can now double click the software and you should see something similar to this:


![Upload FIle](https://user-images.githubusercontent.com/73777608/200693442-590f8793-e20c-4561-b8ff-308e9f047131.PNG)

If your File doesn't exist it will return a error. Once uploaded you can pick a choice by entering the number next to it and clicking enter.

![EnterChoice](https://user-images.githubusercontent.com/73777608/200694514-424d7366-6ae2-47bc-98ff-d7ccb0e8f071.PNG)


# 2) Summary Report

You can select the summary report option by entering 1. This can summarize your data. The summary report will include column values in your dataset and all unique values that have appeared in each column. Before providing the report, you will have to select whether you want to summarize one column or you want a report on all columns:

![Summary_Choice](https://user-images.githubusercontent.com/73777608/203100527-0eb457b1-cea9-4bfc-9c0b-c21893a77cf3.png)

The Specific Value report should then ask you to enter the number next to your choice to select it. The report will look something like this:

![SpecificValueReport](https://user-images.githubusercontent.com/73777608/203101112-3e59ac9c-f9b1-42f2-a93e-cf1c10a30b5b.png)

The Complete Report will do the same above for every column on your dataset.



# 3) Visualize Data

You can select the visualize data option by entering 2. This can visualize your datain any graph you wish. It will ask you for a x and y value for your graph. The x and y will usually be columns. Once entered you have to select which graph you want to visualize in. Your screen should look similar to this:

![VisualizeGraph](https://user-images.githubusercontent.com/73777608/203102446-94e55bfb-9f89-4b64-8ca9-9ba6e63d6b16.png)

This will produce the graph of your choice using matplotlib. Here is an example of a scatterplot:

![Graph](https://user-images.githubusercontent.com/73777608/203102766-4fa7db30-45bc-4338-9545-6201442cbd5b.png)


# 4) Predict Data

You can select the predict data option by entering 3. This can predict your data, given what you want to predict, and what feature do you want to use to predict. 


![Predict1](https://user-images.githubusercontent.com/73777608/203618373-89eb94cf-c222-490c-ba2d-608af4ec9d2a.png)


First you enter what you want to predict, then you enter based on which column do you want to predict that value. For my student database example, lets say I want to predict anyone's reading score, based on their writing score. After entering those values your screen will look similar to this:


![image](https://user-images.githubusercontent.com/73777608/203618519-f18636e0-3d6e-4c27-979f-d527aa1c01b5.png)

Some Machine learning algorithms will do better than others and the best one gets selected. In this case AI 1 gets selected. The test results show how the AI performed on the accuracy test.It shows what it predicted for different values from the dataset and what the actual value was. 

![Predict 3](https://user-images.githubusercontent.com/73777608/203619233-5e8626e4-3c3f-49db-a84a-fa40bf8250d8.png)


Scrolling down from the test results, you can see it asking you to enter a value. Here you enter the value of what you wanted to use to predict and the algorithm will give you a predcited result.

![Predict 4](https://user-images.githubusercontent.com/73777608/203619449-f46881f2-5d85-46bd-9cea-a7a1186237af.png)



# Updates

More instructions will be added to this README and the software soon.
