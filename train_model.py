import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


print("Loading dataset...")
df = pd.read_csv("srilanka_electricity_dataset.csv")

#
def preprocess_data(df):
   
    company_monthly_data = []
    
    # Group by company and month
    for (company, month), group in df.groupby(['company_id', 'month']):
        
        company_size = group['company_size'].iloc[0]
        tariff_category = group['tariff_category'].iloc[0]
        actual_tariff = group['actual_tariff_used'].iloc[0]
        working_days = group['working_days'].iloc[0]
        
       
        total_day_kwh = group['total_company_day_kwh'].iloc[0]
        total_peak_kwh = group['total_company_peak_kwh'].iloc[0]
        total_off_peak_kwh = group['total_company_off_peak_kwh'].iloc[0]
        total_monthly_kwh = group['total_monthly_consumption_kwh'].iloc[0]
        max_demand_kva = group['max_demand_kva'].iloc[0]
        fixed_charge = group['fixed_charge'].iloc[0]
        day_energy_charge = group['day_energy_charge'].iloc[0]
        peak_energy_charge = group['peak_energy_charge'].iloc[0]
        off_peak_energy_charge = group['off_peak_energy_charge'].iloc[0]
        demand_charge = group['demand_charge'].iloc[0]
        total_bill = group['total_bill'].iloc[0]
        
        
        num_machines = len(group)
        total_kw = group['kw'].sum()
        avg_kw = group['kw'].mean()
        max_kw = group['kw'].max()
        avg_power_factor = group['power_factor'].mean()
        
        # average hours 
        avg_day_hours = group['day_hours'].mean()
        avg_peak_hours = group['peak_hours'].mean()
        avg_off_peak_hours = group['off_peak_hours'].mean()
        
        #  machine counts by type
        machine_counts = group['machine_name'].value_counts().to_dict()
        
        # add records for company-month
        record = {
            'company_id': company,
            'month': month,
            'company_size': company_size,
            'tariff_category': tariff_category,
            'actual_tariff': actual_tariff,
            'working_days': working_days,
            'num_machines': num_machines,
            'total_kw': total_kw,
            'avg_kw': avg_kw,
            'max_kw': max_kw,
            'avg_power_factor': avg_power_factor,
            'avg_day_hours': avg_day_hours,
            'avg_peak_hours': avg_peak_hours,
            'avg_off_peak_hours': avg_off_peak_hours,
            'total_day_kwh': total_day_kwh,
            'total_peak_kwh': total_peak_kwh,
            'total_off_peak_kwh': total_off_peak_kwh,
            'total_monthly_kwh': total_monthly_kwh,
            'max_demand_kva': max_demand_kva,
            'fixed_charge': fixed_charge,
            'day_energy_charge': day_energy_charge,
            'peak_energy_charge': peak_energy_charge,
            'off_peak_energy_charge': off_peak_energy_charge,
            'demand_charge': demand_charge,
            'total_bill': total_bill
        }
        
        # Add machine counts
        for machine in machines:
            machine_name = machine["name"]
            record[f'count_{machine_name.replace(" ", "_")}'] = machine_counts.get(machine_name, 0)
        
        company_monthly_data.append(record)
    
    # Create a new dataframe
    return pd.DataFrame(company_monthly_data)

# Define machine types
machines = [
    {"name": "Air Compressor"},
    {"name": "CNC Machine"},
    {"name": "Conveyor Belt"},
    {"name": "Electric Furnace"},
    {"name": "Hydraulic Press"},
    {"name": "Injection Molding Machine"},
    {"name": "Industrial Mixer"},
    {"name": "Lathe Machine"},
    {"name": "Milling Machine"},
    {"name": "Packaging Machine"},
    {"name": "Pump"},
    {"name": "Welding Machine"}
]

print("Processing and aggregating data...")

agg_df = preprocess_data(df)


agg_df[['year', 'month_num']] = agg_df['month'].str.split('-', expand=True)
agg_df['month_num'] = agg_df['month_num'].astype(int)
agg_df['year'] = agg_df['year'].astype(int)

# season (Feb-Mar, Oct-Dec), non-season (Jan-Feb, Apr-Sep)
agg_df['season'] = agg_df['month_num'].apply(
    lambda m: 'Sesson' if m in [2, 3, 10, 11, 12] else 'Non-Sesson'
)


features_to_exclude = ['month', 'fixed_charge', 'day_energy_charge', 'peak_energy_charge', 
                     'off_peak_energy_charge', 'demand_charge', 'total_bill']

X = agg_df.drop(features_to_exclude, axis=1)
y = agg_df['total_bill']

print(f"Feature set shape: {X.shape}")

# data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# numeric and categorical features
numeric_features = [
    'working_days', 'num_machines', 'total_kw', 'avg_kw', 'max_kw', 
    'avg_power_factor', 'avg_day_hours', 'avg_peak_hours', 'avg_off_peak_hours',
    'total_day_kwh', 'total_peak_kwh', 'total_off_peak_kwh', 'total_monthly_kwh',
    'max_demand_kva', 'month_num'
]

# machine count features to numeric features
for machine in machines:
    machine_count_feature = f'count_{machine["name"].replace(" ", "_")}'
    if machine_count_feature in X.columns:
        numeric_features.append(machine_count_feature)

categorical_features = ['company_size', 'tariff_category', 'actual_tariff', 'season', 'year']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# create and train the model
print("Training model...")
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)


print("Evaluating model...")
y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"Mean Absolute Error: Rs. {mae:.2f}")
print(f"Root Mean Squared Error: Rs. {rmse:.2f}")
print(f"R-squared: {r2:.4f}")

# save model
joblib.dump(model, 'srilanka_electricity_predictor.pkl')
print("Model saved as 'srilanka_electricity_predictor.pkl'")



# performance.png
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Bill (Rs.)')
plt.ylabel('Predicted Bill (Rs.)')
plt.title('Actual vs. Predicted Electricity Bills')
plt.savefig('model_performance.png')
plt.close()



# error_distribution.png
plt.figure(figsize=(10, 6))
errors = y_pred - y_test
sns.histplot(errors, kde=True)
plt.xlabel('Prediction Error (Rs.)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.axvline(x=0, color='r', linestyle='--')
plt.savefig('error_distribution.png')
plt.close()




# feature importance png
if hasattr(model['regressor'], 'feature_importances_'):
    # Get feature names after preprocessing
    ohe_features = []
    if hasattr(model['preprocessor'].transformers_[1][1]['onehot'], 'get_feature_names_out'):
        ohe_features = model['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_features)
    else:
        # Rough approximation if method not available
        ohe_features = [f"{cat}_{val}" for cat in categorical_features 
                       for val in agg_df[cat].unique()]
    
    feature_names = numeric_features + list(ohe_features)
    importances = model['regressor'].feature_importances_
    
    # Sort feature importances
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top 20 Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    print("Feature importance plot saved as 'feature_importance.png'")

print("Model training and evaluation complete!")