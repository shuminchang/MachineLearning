import numpy as np
import pandas as pd

data = pd.read_csv('../input/train.csv')

y = data.SalePrice
X = data.drop(['SalePrice'], axis=1)

cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)
       
numerical_cols = [cname for cname in X.columns if X[cname].dtypes in ['int64', 'float64']]
low_cardinality_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and X[cname].dtype == "object"]
my_cols = low_cardinality_cols + numerical_cols
object_cols = [cname for cname in X.columns if X[cname].dtypes in ['object']]

# Numerical columns pretreatment
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_num = pd.DataFrame(my_imputer.fit_transform(X[numerical_cols]))
imputed_X_num.columns = X[numerical_cols].columns

# Categorical columns pretreatment
from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X[object_cols]))
OH_cols_train.index = X[object_cols].index

OH_X_train = pd.concat([imputed_X_num, OH_cols_train], axis=1)

# Prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

model = RandomForestRegressor(n_estimators=500, random_state=0)
scores = -1 * cross_val_score(model, OH_X_train, y, cv=5, scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores.mean())