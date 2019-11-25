from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


def train_model():
    """Function that trains a Multinomial Naives Bayes Classifier. Data is fetched form 
    the processed folder in form of a pandas DataFrame stored in a csv file. This function
    additionaly runs a train-test split. 
    Return a dictionary of the form: 
    {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
           'vect': vect, 'model': model} """

    path_to_data = '/Users/juanpablomejia/Desktop/german_articles/data/processed/df.csv'

    df = pd.read_csv(path_to_data)

    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, ['k_letters']],
                                                        df['articles_cat'], random_state=0)

    vect = CountVectorizer().fit(X_train['k_letters'])
    X_train_vectorized = vect.transform(X_train['k_letters'])

    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vectorized, y_train)

    dic = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
           'vect': vect, 'model': model}

    return(dic)
