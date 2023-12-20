
import pandas as pd
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import chi2
#from sklearn.inspection import plot_partial_dependence

df = pd.read_csv('Diabetes.csv', na_values='?')

df = df.drop(['weight', 'payer_code'], axis=1)  # Drop 'weight' and 'payer_code' columns
df[['diag_1', 'diag_2', 'diag_3']] = df[['diag_1', 'diag_2', 'diag_3']].apply(pd.to_numeric, errors='coerce')
X = df.drop('readmitted', axis=1) 
#X2 = df.drop('readmitted', axis=1)   
y = df['readmitted'].astype(int)

#categorical variables
categorical_cols = ['race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
                    'medical_specialty','max_glu_serum','A1Cresult','metformin','repaglinide','nateglinide','chlorpropamide',
                    'glimepiride','acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone','acarbose',
                    'miglitol','troglitazone','tolazamide','examide','citoglipton','insulin','glyburide.metformin',
                    'glipizide.metformin','glimepiride.pioglitazone','metformin.rosiglitazone','metformin.pioglitazone',
                    'change','diabetesMed','diag_1_desc','diag_2_desc','diag_3_desc']

encoder = LabelEncoder()

for col in categorical_cols:
    X[col] = encoder.fit_transform(X[col])
X2=X

imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

#print(X)
#One-hot encode
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_cols)], remainder='passthrough')
X = ct.fit_transform(X)
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.2, random_state=43)

#Scaling
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#RF
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

#Accuracy and test
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(f"model score on training data: {rf_model.score(X_train, y_train)}")
print(f"model score on testing data: {rf_model.score(X_test, y_test)}")

# feature importance
#result = permutation_importance(rf_model, X_test.toarray(), y_test, n_repeats=1, random_state=43)
#importances = result.importances_mean

#for feature, importance in zip(X.columns, importances):
#    print(f"{feature}: {importance}")

#print(X_test)
#print(y_test)
# Create an explainer dashboard
#explainer = ClassifierExplainer(rf_model, X_test2, y_test2)
#explainer.run()
#ExplainerDashboard(explainer,shap_interaction=False,no_permutations=True,plot_sample=1000).run()


