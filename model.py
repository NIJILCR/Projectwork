# Income orediction model
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
li=pd.read_csv("preprocessed_dataset.csv")
X=li.drop(['avgCompanyPosDuration'],axis=1)
y=li['avgCompanyPosDuration']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.20)
from sklearn.ensemble import RandomForestRegressor
r = RandomForestRegressor(n_estimators = 100, random_state = 42)
model=r.fit(X_train,y_train)
#Saving the model to disk
pickle.dump(r,open('model.pkl','wb') )

