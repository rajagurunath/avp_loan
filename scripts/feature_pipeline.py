from imblearn.ensemble import BalanceCascade
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import path
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import FunctionTransformer


pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
DATA_DIR=path.Path("../data/")
ARTIFACT_DIR=path.Path("../artifacts/")\

train=pd.read_csv(DATA_DIR+"train.csv")
test=pd.read_csv(DATA_DIR+"test.csv")
train.head()

def time_function(df,columns):
    for col in columns:
        df['{}_transformed'.format(col)]=pd.to_datetime(train[col])
    return df
# train=time_function(train,['origination_date','first_payment_date'])\\

def make_time_features(df,columns):
    df=time_function(df,columns)
    for col in columns:
        df[f'{col}_day']=df[col+"_transformed"].dt.day#.value_counts()
        df[f'{col}_week']=df[col+"_transformed"].dt.week#.value_counts()
        df[f'{col}_month']=df[col+"_transformed"].dt.month#.value_counts()
    df['day_difference']=(df['first_payment_date_transformed']-df['origination_date_transformed']).dt.days#.value_counts()
    for col in columns:
        df=df.drop(col,axis=1)
        df=df.drop(col+"_transformed",axis=1)
    return df



numerical_columns=train.columns[1:-1].tolist()
categorical_features=['source',"financial_institution","loan_purpose"]
time_columns=['origination_date','first_payment_date']
bin_features=['interest_rate','debt_to_income_ratio','borrower_credit_score','insurance_percent','co-borrower_credit_score']
for cat in categorical_features:numerical_columns.remove(cat)
for tcat in time_columns:numerical_columns.remove(tcat)
for bcat in bin_features:numerical_columns.remove(bcat)


transformed_cols=[f"{col}_transformed" for col in time_columns]

timefeat=FunctionTransformer(func=make_time_features,kw_args=dict(columns=time_columns),validate=False)
# droptimefeat=FunctionTransformer(func=drop_time_columns,kw_args=dict(columns=time_columns),validate=False)

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

bin_transformer=Pipeline(
        steps=[
            ("bin_transformer",KBinsDiscretizer())
        ]

)
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
print(time_columns)
print(time_columns+transformed_cols)
preprocessor = ColumnTransformer(
    transformers=[
        ('timefeat',timefeat,time_columns),
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_features),
        ('bin',bin_transformer,bin_features),
        
    ])

train1=preprocessor.fit_transform(train.iloc[:,:-1])
test1=preprocessor.transform(test)

