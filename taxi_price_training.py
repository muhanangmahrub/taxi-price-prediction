import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from google.cloud import storage
from joblib import dump
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

storage_client = storage.Client()
bucket = storage_client.bucket("taxi-price-bucket")

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def preprocess_data(df, target):
    df = df.dropna(subset=[target])

    numerical_features = ['Trip_Distance_km', 'Passenger_Count', 'Base_Fare', 'Per_Km_Rate', 'Per_Minute_Rate', 'Trip_Duration_Minutes']
    categorical_features = ['Time_of_Day', 'Day_of_Week', 'Traffic_Conditions', 'Weather']

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    X = df.drop(columns=[target])
    y = df[target]

    X_preprocessed = pipeline.fit_transform(X)

    preprocessed_feature_names = (
        numerical_features +
        list(pipeline.named_steps['preprocessor']
             .named_transformers_['cat']
             .named_steps['onehot']
             .get_feature_names(categorical_features))
    )

    X_preprocessed_df = pd.DataFrame(X_preprocessed, columns=preprocessed_feature_names)
    return X_preprocessed_df, y

def train_model(x_train, y_train):
    model = RandomForestRegressor()
    pipeline = make_pipeline(model)
    pipeline.fit(x_train, y_train)
    return pipeline

def save_model_artifact(pipeline):
    artifact_name = 'model.joblib'
    dump(pipeline, artifact_name)
    model_artifact = bucket.blob('taxi-price-artifact/' + artifact_name)
    model_artifact.upload_from_filename(artifact_name)

def main():
    filename = 'gs://taxi-price-bucket/taxi_trip_pricing.csv'
    df = load_data(filename)
    X, y = preprocess_data(df, 'Trip_Price')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pipeline = train_model(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    save_model_artifact(pipeline)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print('RMSE', rmse)

if __name__ == '__main__':
    main()