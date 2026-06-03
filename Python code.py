import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score)
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import joblib

import statsmodels.api as sm
from statsmodels.tools import add_constant

from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv(r"/content/burn.csv")
df

df.info()

df.describe()

df.isnull().sum()

df.duplicated().sum()

for col in df.select_dtypes(include=np.number).columns:
    print(col, df[col].min(), df[col].max())

df['age_group'] = pd.cut(df['AGE'], bins=[0,30,50,70,100],
                         labels=['<30','30-50','50-70','70+'])
df

df.groupby('GENDER')['AGE'].mean()

df.groupby('GENDER')['DEATH'].mean()

df.groupby('RACEC')['DEATH'].mean()

df.groupby('age_group')['DEATH'].mean()

df.groupby('age_group')['TBSA'].mean()

df.groupby('age_group')['INH_INJ'].mean()

df.groupby('age_group')['FLAME'].mean()

df.groupby(['GENDER','age_group'])['DEATH'].mean()

df.groupby(['GENDER','age_group'])['TBSA'].mean()

df.groupby(['GENDER','age_group'])['FLAME'].mean()

df.groupby(['GENDER','age_group'])['INH_INJ'].mean()

pd.crosstab(df['GENDER'], df['DEATH'])

pd.crosstab(df['age_group'], df['DEATH'])

pd.crosstab(df['DEATH'], df['INH_INJ'], margins=True)

sns.histplot(df['TBSA'], kde=True); plt.title('TBSA distribution'); plt.show()
sns.boxplot(x='DEATH', y='TBSA', data=df); plt.title('TBSA by death'); plt.show()

sns.histplot(df['AGE'], kde=True); plt.title('Age distribution'); plt.show()
sns.countplot(x='INH_INJ', hue='DEATH', data=df); plt.title('Inhalation injury vs death'); plt.show()

plt.figure(figsize=(6,5))
sns.heatmap(df[['AGE','TBSA','DEATH']].corr(), annot=True, cmap='coolwarm'); plt.show()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

y = df["DEATH"]   # target: 1 = died, 0 = survived
X = df.drop("DEATH", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import matplotlib.pyplot as plt

z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))

plt.plot(z, sigmoid)
plt.title("Sigmoid Function")
plt.xlabel("z (linear score)")
plt.ylabel("Probability")
plt.grid(True)
plt.show()

import sys, traceback
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("\nDtype counts in X_train:\n", X_train.dtypes.value_counts())
print("\nAny object/category cols:", X_train.select_dtypes(include=['object','category']).columns.tolist())
print("\nAny NaNs in X_train per column:\n", X_train.isna().sum().loc[lambda s: s>0])
print("\nAny infinite values anywhere?", np.isinf(X_train.select_dtypes(include=[np.number])).values.any())

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object','category','bool']).columns.tolist()
print("\nNumeric cols:", num_cols)
print("Categorical cols:", cat_cols)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    remainder='drop'
)

y_pred = best_model.predict(X_test)

print("Objective: Predict survival after burn injury")
print("Accuracy on test data:", best_model.score(X_test, y_test))

print(best_model.named_steps.keys())

pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(solver='saga', max_iter=5000))
])

feature_names = list(best_model.named_steps['preprocessor'].transformers_[0][2]) + \
                list(best_model.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(cat_cols))

coeff_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": best_model.named_steps['log_reg'].coef_[0]
})
coeff_df["Odds Ratio"] = np.exp(coeff_df["Coefficient"])
print(coeff_df.sort_values(by="Odds Ratio", ascending=False))

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

y_prob = best_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

import statsmodels.api as sm

X_train_sm = X_train.copy()
X_train_sm = pd.get_dummies(X_train_sm, columns=['age_group'], drop_first=True)

for col in X_train_sm.select_dtypes(include='bool').columns:
    X_train_sm[col] = X_train_sm[col].astype(int)

X_train_sm = X_train_sm.drop(['ID', 'AGE'], axis=1)
X_train_sm = sm.add_constant(X_train_sm)
logit_model = sm.Logit(y_train, X_train_sm).fit()
print(logit_model.summary())

residuals = y_train - logit_model.predict(X_train_sm)
plt.scatter(logit_model.predict(X_train_sm), residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted probability")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()