import numpy as np
import pandas as pd
from src.models.train_model import train_model

dic = train_model()

model = dic['model']
vect = dic['vect']


def predict_class(row):
    row['class_cat'] = model.predict(vect.transform([row['feat_name']]))[0]
    probabilities = list(model.predict_proba(
        vect.transform([row['feat_name']])))[0]
    row['probabilities'] = round(max(probabilities), 2)
    if row['class_cat'] == 0:
        row['class'] = 'Das'
    if row['class_cat'] == 1:
        row['class'] = 'Der'
    if row['class_cat'] == 2:
        row['class'] = 'Die'
    return row


def gen_df_results():

    feat_value = model.coef_[0]
    order_of_importance = (-feat_value).argsort()
    feat_names = np.array(vect.get_feature_names())

    dic_results = {'feat_name': feat_names[order_of_importance],
                   'feat_value': feat_value[order_of_importance]}
    df_results = pd.DataFrame(dic_results)
    df_results = df_results.apply(predict_class, axis=1)

    return df_results
