import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_validate,KFold
import path
from typing import Callable

DATA_DIR=path.Path("../data/")
ARTIFACT_DIR=path.Path("../artifacts/")
def evaluate(y_test,pred):
    return f1_score(y_test, pred)# average='macro')

def helper_cross_validate(X:pd.DataFrame,y:pd.Series,train_model:Callable):
    kf=KFold(n_splits=10)
    score_list=[]
    for train_index,test_index in kf.split(X):
      X_train,X_test=X.iloc[train_index],X.iloc[test_index]
      y_train,y_test=y.iloc[train_index],y.iloc[test_index]
    #   if pass_val:
    #     model=train_model(model,X_train,y_train,X_test,y_test)
    #   else:
    #     model=train_model(model,X_train,y_train)
      
      model=train_model(X_train,y_train,X_test,y_test)
      pred=model.predict(X_test)
      score=evaluate(y_test,pred)
      print("score:",score)
      score_list.append(score)
      #pd.Series(rf.feature_importances_).plot()
      #plt.show()  
      #ploting
    #plt.figure(figsize=(20,15))
    plt.bar(range(10),score_list)
    plt.show()
  
