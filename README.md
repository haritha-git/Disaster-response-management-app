# Disaster Response Pipeline Project
This project is a web app where an emergency worker can input messages from real-life disasters and get classification results highlighted in respective categories.
## Table of Contents:
1.[Instructions](#Instructions)
2.[Dataset Visuals](#Visuals)
3.[Authors](#Authors)
<a name="Instructions"></a>
## Instructions:
a. Clone this GIT repository:
```
git clone https://github.com/haritha-git/app.git
```
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
*[G. Haritha](https://github.com/haritha-git)
