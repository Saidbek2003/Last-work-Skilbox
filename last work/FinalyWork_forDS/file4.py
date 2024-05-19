import dill
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def transform_data(data):
    #ohe = OneHotEncoder(sparse=False)
    with open('OHE.pickle', 'rb') as file:
        ohe = dill.load(file)

    print('ok',data.info())
    ohe_columns = ohe.transform(data)
    #print(ohe_columns)
    #print(ohe.get_feature_names_out())
    df_prepared = pd.DataFrame(ohe_columns, columns=list(ohe.get_feature_names_out()))
    return df_prepared


def load_data():

    data = pd.read_csv('df_test.csv')

    data=data.drop(columns=['session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number'])
    return data


def predict_on_data(data):
    model = joblib.load('model.pkl')
    return pd.DataFrame(model['model'].predict(data), columns=['target'])


def main():

    df = transform_data(load_data())
    predict = predict_on_data(df)
    print(predict)
    print(predict.value_counts())
    predict.to_csv('predict_test.csv', index_label=False, sep=',')


if __name__ == '__main__':
    main()