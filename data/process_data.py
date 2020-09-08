import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Function load dataframe from messages and categories csv files
    INPUT:messages.csv and categories.csv files
    OUTPUT:Dataframe of merged input datasets
    """
    #load messages dataset
    messages = pd.read_csv(messages_filepath)
    #load categories dataset
    categories = pd.read_csv(categories_filepath)
    #merge datasets using the common id
    df = messages.merge(categories, on='id')
    print("Data is loading");
    print(df.head(2));
    return df


def clean_data(df):
    """
    Function to clean and obtain features from categories dataframe
    INPUT: Dataframe to be cleaned
    OUTPUT: cleaned and wrangled Dataframe
    """
    print("Data is cleaning");
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # using this row to extract a list of new column names for categories.
    colnames = []
    for entry in row:
        colnames.append(entry[:-2])
    category_colnames = colnames
    # rename the columns of `categories`
    categories.columns = category_colnames
    print(category_colnames)
    #convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    #Replace categories column in df with new category columns.
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    print(df.head())
    return df

def save_data(df, database_filename):
    """
    saves Dataframe to Database in the same folder
    INPUT: dataframe to be converted
    OUTPUT: Database
    """
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('Disasters', engine,if_exists='replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
