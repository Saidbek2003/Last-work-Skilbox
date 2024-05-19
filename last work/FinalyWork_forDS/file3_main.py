import dill
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import json

app = FastAPI()
#Обычный модель
model = joblib.load('model.pkl')
#Модел с исползованый пайплан
model_pip = joblib.load('p_model.pkl')
# ненужные параметры
col_d = ['session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number'] 
#Структура форма
class Form(BaseModel):
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: str
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


#Структура ответ
class Prediction(BaseModel):
    id: str
    Result: int
#Get запрос для тест
@app.get('/status')
def status():
    return "Server rabotaet!!! "

#Get запрос для информация о модел
@app.get('/version')
def version():
    return model['metadata']

#Обработка обычный модель
@app.post('/predict', response_model=Prediction)
def predict(form: Form):

    df = pd.DataFrame.from_dict([form.dict()])
    df=df.drop(columns=col_d)
    #print(' df=',df)
    with open('OHE.pickle', 'rb') as file:
        ohe = dill.load(file)
    ohe_columns = ohe.transform(df)
    df_prepared = pd.DataFrame(ohe_columns, columns=list(ohe.get_feature_names_out()))
    y = model['model'].predict(df_prepared)

    return {
        'id': form.session_id,
        'Result': y[0]
    }

#Обработка модель с пайпланом
@app.post('/predict_pip', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    df = df.drop(columns=col_d)
    # print(' df=',df)
    y = model_pip['model'].predict(df)

    return {
        'id': form.session_id,
        'Result': y[0]
    }

#Локалный тест модель с пайпланом
def main_pip():
    with open('json_test.json') as fin:
        form = json.load(fin)
        session_id=form[0]["session_id"]
        df = pd.DataFrame.from_dict(form)
        #print(' df ',df)

        df = df.drop(columns=col_d)
        #print(' df ', df)
        y = model_pip['model'].predict(df)

        print(f'{session_id}:{y[0]}')

#Локалный тест модель
def main():
    with open('json_test.json') as fin:
        form = json.load(fin)
        session_id=form[0]["session_id"]

        df = pd.DataFrame.from_dict(form)
        df = df.drop(columns=col_d)
        #print(' df ',df)
        with open('ohe.pickle', 'rb') as file:
            ohe = dill.load(file)
        ohe_columns = ohe.transform(df)
        df_prepared = pd.DataFrame(ohe_columns, columns=list(ohe.get_feature_names_out()))
        y = model['model'].predict(df_prepared)

        print(f'{session_id}:{y[0]}')


#if __name__ == '__main__':
#    main()
#uvicorn main:app --host 0.0.0.0 --port 8000 --reload