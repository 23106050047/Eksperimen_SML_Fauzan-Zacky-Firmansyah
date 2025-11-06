import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder


def cap_outliers(df, feature, multiplier=1.5):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    df[feature] = df[feature].clip(lower=lower_bound, upper=upper_bound)
    return df


def automate_preprocessing_pipeline(data_path, save_csv_path):
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    df = pd.read_csv(data_path)

    duplicates_before = df.duplicated().sum()
    df.drop_duplicates(inplace=True)

    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['float64', 'int64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)

    numerical_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    categorical_features = ['gender', 'smoking_history']
    binary_features = ['hypertension', 'heart_disease']
    target_col = 'diabetes'

    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    for feature in numerical_features:
        df = cap_outliers(df, feature, multiplier=1.5)
    le = LabelEncoder()
    for col in categorical_features:
        df[col] = le.fit_transform(df[col])
        
    feature_columns = numerical_features + categorical_features + binary_features
    if target_col in df.columns:
        final_columns = feature_columns + [target_col]
        df = df[final_columns]
    else:
        df = df[feature_columns]

    df.to_csv(save_csv_path, index=False, header=True)
    print(f"✅ Preprocessing selesai!")
    print(f"   - File disimpan di: {save_csv_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    default_data_path = os.path.join(project_dir, "diabetes_prediction_dataset_raw.csv")
    default_save_path = os.path.join(script_dir, "diabetes_prediction_dataset_preprocessing.csv")
    
    if len(sys.argv) >= 2:
        data_path = sys.argv[1]
    else:
        data_path = default_data_path
    
    if len(sys.argv) >= 3:
        save_path = sys.argv[2]
    else:
        save_path = default_save_path
    
    automate_preprocessing_pipeline(
        data_path=data_path,
        save_csv_path=save_path
    )
