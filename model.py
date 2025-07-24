import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv(r"C:\Users\Rakesh R\3D Objects\project\customer_shopping_data.csv")
df['invoice_date'] = pd.to_datetime(df['invoice_date'], dayfirst=True)
df = df[df['invoice_date'] > "01-01-2021"]
df['price'] = df['price'].astype(float)
df = df.drop_duplicates(subset=['customer_id', 'invoice_date'])

frequency = df.groupby('customer_id')['invoice_no'].nunique().reset_index()
frequency.columns = ['customer_id', 'Frequency']

latest_date = df['invoice_date'].max()
recency = df.groupby('customer_id').agg({'invoice_date': lambda x: x.max()}).reset_index()
recency['Recency'] = (latest_date - recency['invoice_date']).dt.days

aov = df.groupby('customer_id').agg({'price': 'sum', 'invoice_no': 'nunique'}).reset_index()
aov['AOV'] = aov['price'] / aov['invoice_no']
aov = aov[['customer_id', 'AOV']]

features = recency.merge(frequency, on='customer_id').merge(aov, on='customer_id')
features['LVT'] = features['AOV'] * features['Frequency']

X = features[['Recency', 'Frequency', 'AOV']]
y = features['LVT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

plt.figure(figsize=(6, 5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual LTV")
plt.ylabel("Predicted LTV")
plt.title("Actual vs Predicted LTV")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.tight_layout()
plt.show()

new_data = pd.DataFrame({
    'Recency': [15],
    'Frequency': [3],
    'AOV': [350.0]
})
ltv_prediction = model.predict(new_data)
print(f"Predicted LTV for new customer: {ltv_prediction[0]:.2f}")

features['Predicted_LTV'] = model.predict(features[['Recency', 'Frequency', 'AOV']])

def segment_ltv(ltv):
    if ltv > 1000:
        return 'High'
    elif ltv > 500:
        return 'Medium'
    else:
        return 'Low'

features['LTV_Segment'] = features['Predicted_LTV'].apply(segment_ltv)

plt.figure(figsize=(8, 5))
sns.histplot(features['Predicted_LTV'], bins=30, kde=True)
plt.title("Distribution of Predicted Customer LTV")
plt.xlabel("Predicted LTV")
plt.ylabel("Customer Count")
plt.tight_layout()
plt.show()

feature_importances = model.feature_importances_
features_list = ['Recency', 'Frequency', 'AOV']

plt.figure(figsize=(6, 4))
plt.barh(features_list, feature_importances)
plt.xlabel("Importance")
plt.title("Feature Importance for LTV Prediction")
plt.tight_layout()
plt.show()

save_dir = r"C:\Users\Rakesh R\3D Objects\project\lvt_prediction"
os.makedirs(save_dir, exist_ok=True)

csv_path = os.path.join(save_dir, "customer_ltv_features.csv")
model_path = os.path.join(save_dir, "ltv_model.pkl")

features.to_csv(csv_path, index=False)
joblib.dump(model, model_path)

print(f"CSV saved at: {csv_path}")
print(f"Model saved at: {model_path}")
