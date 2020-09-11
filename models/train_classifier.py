import sys
import nltk
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

def load_data(database_filepath):
    """Load the filepath and return the data
       INPUT: database_filepath
       OUTPUT: Dataframe
               X: message column of dataframe as the input features
               y: categories of the dataframe for ML classification
               cate
    """
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('Disasters', con=engine) # is table always called this?
    print(df.head())
    X = df['message']
    y = df[df.columns[4:]]
    added = pd.get_dummies(df[['genre']])
    y = pd.concat([y, added], axis=1)
    return X, y

def tokenize(text):
    """
    Function to tokenize and lemmatize the given text
    INPUT: text message
    OUTPUT:clean_tokens is a processed message
    """
    #remove punctuation and url links
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    #tokenize text
    tokens = word_tokenize(text)
    #remove stopwords
    stop_words = stopwords.words("english")
    tokens = [token for token in  tokens if token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """Return Grid Search model with pipeline and Classifier"""
    # Create pipeline with Classifier
    moc = MultiOutputClassifier(RandomForestClassifier())
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
        ])

    parameters = {'clf__estimator__criterion': ['entropy'],
              'clf__estimator__max_depth': [10, 50, None],
             # 'clf__estimator__min_samples_leaf':[2, 5, 10],
              'clf__estimator__n_estimators': [10,50,100]}

    cv=GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=1)
    return cv
def precision_recall_fscore(y_test,y_pred):
    """
    Function using  precision_recall_fscore_support provide better report
    Input: y_test and y_pred
    Output: prints the precision,recall,fscore report of each category

    """
    report = pd.DataFrame(columns=['Category', 'f_score', 'precision', 'recall'])

    for i,col in enumerate(y_test):
        precision, recall, f_score, support = precision_recall_fscore_support(y_test[col], y_pred[:,i], average='weighted')
        report.set_value(i, 'Category', col)
        report.set_value(i, 'f_score', f_score)
        report.set_value(i, 'precision', precision)
        report.set_value(i, 'recall', recall)
    print('simply the average results are:')
    print('Mean f_score:', report['f_score'].mean())
    print('Mean precision:', report['precision'].mean())
    print('Mean recall:', report['recall'].mean())
    return report

def save_model(model, model_filepath):
    """Save model as pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    """Load the data, run the model and save model"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y= load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train.as_matrix(), y_train.as_matrix())
        y_pred1 = model.predict(X_test)

        print('Evaluating model...')
        report1 =precision_recall_fscore(y_test, y_pred1)
        print(report1)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
