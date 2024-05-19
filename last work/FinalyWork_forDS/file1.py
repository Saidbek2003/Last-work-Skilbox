import dill
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Функция выделения целевых действий
def prepare_target_df():
    print('Start preparing target df...')
    # Загрузка исходных датасетов
    df_hits = pd.read_csv('ga_hits.csv', low_memory=False)
    df_session = pd.read_csv('ga_sessions.csv', low_memory=False)

    # Выделение целевых переменных
    target = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click',
              'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click',
              'sub_submit_success', 'sub_car_request_submit_click']
    df_target = df_hits[df_hits['event_action'].isin(target)]
    df_session['target'] = df_session['session_id'].isin(df_target['session_id']).astype(int)

    # Откидываем лишние столбцы
    df_session = df_session.drop(columns=['session_id', 'client_id', 'visit_date', 'visit_time', 'visit_number'])

    print('End preparing target df')
    return df_session
# Преобразование типов
def prepare_types(df):
    print('Start preparing data types...')
    feature_to_str = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_adcontent', 'utm_keyword', 'device_category',
                      'device_os', 'device_brand', 'device_model', 'device_screen_resolution',
                      'device_browser', 'geo_country', 'geo_city']
    for x in feature_to_str:
        df[x] = df[x].dropna().astype('str')
    print('End preparing data types')
    return df
# Обработка пропусков
def fill_nan(df):
    print('Start filling NaN...')
    # Пропуски в device_model
    df['device_model'] = df['device_model'].fillna('noname')

    # Пропуски в utm_keyword
    df['utm_keyword'] = df['utm_keyword'].fillna('other')

    # Пропуски в device_brand
    df.loc[(df['device_category'] == 'desktop') & df['device_brand'].isna() & df['device_os'].isna(),
           ['device_os', 'device_brand']] = 'other'

    df.loc[(df['device_category'] == 'desktop') & df['device_brand'].isna() &
           ((df['device_os'] == 'Windows') |
            (df['device_os'] == 'Linux') |
            (df['device_os'] == 'Chrome OS') |
            (df['device_os'] == '(not set)')), 'device_brand'] = 'PC'

    df.loc[df['device_os'] == 'Macintosh', 'device_brand'] = 'Apple'

    df.loc[(df['device_brand'] == 'Apple') & (df['device_category'] == 'desktop') & df['device_os'].isna(),
           'device_os'] = 'Macintosh'

    df.loc[(df['device_category'] == 'desktop') & df['device_os'].isna(), 'device_os'] = 'Linux'

    df.loc[(df['device_brand'] == 'Apple') & (df['device_category'] == 'desktop') & df['device_os'].isna(),
           'device_os'] = 'iOS'

    df.loc[(df['device_category'] == 'mobile') & df['device_os'].isna(), 'device_os'] = 'Android'

    df.loc[(df['device_brand'] == 'Apple') & (df['device_category'] == 'tablet') & df['device_os'].isna(),
           'device_os'] = 'iOS'

    df.loc[df['device_os'].isna() & (df['device_category'] == 'tablet'), 'device_os'] = 'Android'

    # Пропуски в utm_adcontent
    df['utm_adcontent'] = df['utm_adcontent'].fillna('other')

    # Пропуски в utm_campaign
    df['utm_campaign'] = df['utm_campaign'].fillna('other')

    # Остальные пропуски
    df = df.dropna()
    print('End filling NaN')
    return df

def org_short(df):
    print('Start org_short...')


    # Сделаем составной датасет где количество положительных событий не сильно отличалось от количества нецелевых
    n_event = len(df[df['target'] == 1])
    df_short = pd.concat([df[df['target'] == 1],
                          df[df['target'] == 0].sample(n=2*n_event, random_state=12)]).reset_index(drop=True)



    print('End org_short')
    return df_short






def main():

    print('Avto arenda model_pipe...')
    df=org_short(fill_nan(prepare_target_df()))

    X = df.drop('target', axis=1)
    y = df['target']

    numerical = make_column_selector(dtype_include=['int64', 'float64'])
    categorical = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
     ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
     ])


    preprocessor = ColumnTransformer(transformers=[

        ('numerical', numerical_transformer, numerical),
        ('categorical', categorical_transformer, categorical)
       ])

    models = (DecisionTreeClassifier()
              ,LogisticRegression(solver='liblinear')
              ,RandomForestClassifier(bootstrap=False, max_depth=100,
                                              min_samples_leaf=2, n_jobs=-1,
                                              random_state=12)
              #,MLPClassifier()
              #,SVC
               )

    best_score = .0
    best_pipe = None

    for model in models:
        pipe = Pipeline(steps=[

        ('preprocessor', preprocessor),
        ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    best_pipe.fit(X, y)
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')
    #print(best_pipe)

    with open('p_model.pkl', 'wb') as f:
        dill.dump({

        'model': best_pipe,

        'metadata': {

            'name': 'Avto arenda model pip',

            'author': 'Muhammadjonov Sayidbek',

            'version': 1,

            'date': datetime.now(),

            'type': type(best_pipe.named_steps["classifier"]).__name__,

            'accuracy': best_score

        }

    }, f)



if __name__ == '__main__':
    main()
    #df = org_short(fill_nan(prepare_target_df()))
    #df.info()