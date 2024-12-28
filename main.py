import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.multivariate.manova import MANOVA
from sklearn.compose import ColumnTransformer
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data = pd.DataFrame({
    'Category1': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Category2': ['X', 'X', 'Y', 'Y', 'X', 'Y', 'X', 'Y'],
    'Covariate1': [5.1, 6.2, 7.3, 5.5, 6.8, 7.9, 5.3, 6.6],
    'Covariate2': [1.2, 2.4, 3.5, 2.1, 3.3, 2.2, 1.5, 2.8],
    'Dependent1': [3.2, 2.9, 3.7, 3.0, 2.8, 3.1, 3.4, 2.7],
    'Dependent2': [8.0, 7.9, 8.1, 8.3, 7.8, 8.4, 8.2, 7.7],
    'Dependent3': [15.3, 14.8, 15.0, 14.5, 15.2, 14.9, 15.1, 14.7],
    'Dependent4': [20.1, 21.4, 20.3, 21.0, 20.5, 21.2, 20.2, 21.1],
    'Dependent5': [12.3, 12.0, 12.5, 12.4, 12.1, 12.6, 12.2, 12.7]
})

X = data[['Category1', 'Category2', 'Covariate1', 'Covariate2']]
Y = data[['Dependent1', 'Dependent2', 'Dependent3', 'Dependent4', 'Dependent5']]

categorical_columns = ['Category1', 'Category2']
numerical_columns = ['Covariate1', 'Covariate2']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ])

X_transformed = preprocessor.fit_transform(X)
X_transformed_df = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())

data_for_mancova = pd.concat([X_transformed_df, Y], axis=1)

dependent_vars = " + ".join(Y.columns)
independent_vars = " + ".join(preprocessor.get_feature_names_out())

formula = f'{dependent_vars} ~ {independent_vars}'
mancova = MANOVA.from_formula(formula, data=data_for_mancova)
mancova_results = mancova.mv_test()

print(mancova_results)

dependent_variable = 'Dependent1'
pairwise_results = pairwise_tukeyhsd(data[dependent_variable], data['Category1'])
print(pairwise_results)
