import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and categories and merge datasets on id
    
    Args:
    messages_filepath: string. Filepath for messages dataset.
    categories_filepath: string. Filepath for categories dataset.
       
    Retruns:
    df: dataframe. Merged dataframe containing content of messages and categories datasets
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = pd.merge(messages, categories, on="id")
    
    return df

def clean_data(df):
    
    '''
    Clean dataframe
    
    Args:
    df: dataframe. Merged df containing messages and categories df.
       
    Returns:
    df: dataframe. Cleaned df.
    '''
    
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df.categories.str.split(';',expand=True))
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split('-').apply (lambda x: x[0])

    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.split('-').apply(lambda x:x[1])  
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop('categories', 1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # Remove rows with a related value of 2
    df = df[df['related'] != 2]
    
    return df

def save_data(df, database_filename):
    '''Save df into  SQLite database.
    
    Args:
    df: dataframe. Cleanded df containing cleaned version of merged message and 
    categories data
    
    database_filename: string. Filename for output database
       
    Returns:
    None
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages1', engine, index=False, if_exists='replace')
    

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