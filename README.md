# Disaster Response Pipeline Project
We all know that following a disaster typically gets millions of messages to the disaster response organizations. As different organizations take care of specific problems like water, Hospitals, Aid related, food. The incoming messages are sorted into categories corresponding to organizations, speed up aid, and contribute to a more efficient distribution of people and other resources.

This project is a web app where an emergency worker of the organization can input messages from real-life disasters and get classification results highlighted in respective categories, which could help to speed up their aid efforts.
## Table of Contents:
1.[Libraries](#Libraries)

2.[File Description](#file)

2.[Instructions](#Instructions)

3.[Dataset Visuals](#Visuals)

4.[Authors](#Authors)

5.[Acknowledgements](#Acknowledgements)

<a name="Libraries"></a>
## Libraries:
* pandas

* numpy

* sqlalchemy

* matplotlib

* plotly

* NLTK

* sklearn

* joblib

* flask

<a name="file"></a>
## File Description
1. data

In a Python script, process_data.py is a data cleaning pipeline that:
  - Loads the messages and categories datasets
  - Merges the two datasets
  - Cleans the data
  - Stores it in a SQLite database

2. ML Pipeline

In a Python script, train_classifier.py, write a machine learning pipeline that:
   - Loads data from the SQLite database
   - Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file

3. Flask Web App
In a python script run.py, to run web application, templates for web page
   - add visualizations from test dataset
   - Input message is categorized among 36 categories.
<a name="Instructions"></a>
## Instructions:
a. Clone this GIT repository:
git clone [https://github.com/haritha-git/app.git](https://github.com/haritha-git/app.git)

b. Executing Program:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:3001/

c. Web page working
1. Enter the message inside the text box, click **Classify Message**.
2. You can find the message categorized, as highlighted in green.

<a name="Visuals"></a>
### Dataset Visuals:
These are the Visuals in the web app based on the training dataset provided by Figure Eight.
![genres](https://github.com/haritha-git/app/blob/master/Visuals/genres.png)

![categories](https://github.com/haritha-git/app/blob/master/Visuals/categories.png)

<a name = 'Authors'></a>
## Authors:
   [G. Haritha](https://github.com/haritha-git)
<a name='Acknowledgements'></a>
## Acknowledgements
   - Udacity nano degree program
   - Figure Eight for provding the data used by this project.
