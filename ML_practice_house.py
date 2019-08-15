# Pipeline with missing values
data = pd.read_csv('../input/train.csv')

y = data.SalePrice
X = data.drop(['SalePrice'], axis=1)

numerical_cols = [cname for cname in X.columns if X[cname].dtypes in ['int64', 'float64']]
categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and X[cname].dtypes == 'object']
my_cols = numerical_cols + categorical_cols

X_train = X[my_cols].copy()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=500, random_state=0)

from sklearn.metrics import mean_absolute_error

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', model)
                            ])

from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(my_pipeline, X_train, y, cv=5, scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores.mean())

# Pipeline without missing values
data = pd.read_csv('../input/train.csv')

y = data.SalePrice
X = data.drop(['SalePrice'], axis=1)

cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
X.drop(cols_with_missing, axis=1, inplace=True)

numerical_cols = [cname for cname in X.columns if X[cname].dtypes in ['int64', 'float64']]
categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and X[cname].dtypes == 'object']
my_cols = numerical_cols + categorical_cols

X_train = X[my_cols].copy()

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=500, random_state=0)

from sklearn.metrics import mean_absolute_error

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', model)
                            ])

from sklearn.model_selection import cross_val_score

scores = -1 * cross_val_score(my_pipeline, X_train, y, cv=5, scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores.mean())

# Not Pipeline without missing values
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

# XGBRegressor without missing values with Cross Validation
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and 
                        X[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)

from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score

model = XGBRegressor(n_estimators=500, learning_rate=0.1)
scores = -1 * cross_val_score(model, X_train, y, cv=5, scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
print("MAE scores_mean:\n", scores.mean())
