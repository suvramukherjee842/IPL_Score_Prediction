import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# Load the dataset
file_path = 'matches.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found in working directory.")

df = pd.read_csv(file_path)

# Drop rows where target_runs or target_overs is missing (incomplete or abandoned matches)
df = df.dropna(subset=['target_runs', 'target_overs'])

# Fill missing city values with 'Unknown'
df['city'] = df['city'].fillna('Unknown')

# Feature selection based on available columns
features = [
    'venue', 'team1', 'team2', 'toss_winner', 'toss_decision',
    'city', 'season', 'target_overs'
]
target = 'target_runs'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['target_overs']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [
            'venue', 'team1', 'team2', 'toss_winner', 'toss_decision', 'city', 'season'
        ])
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10]
}
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Save the model
joblib.dump(best_model, 'ipl_score_predictor.pkl')
print("Model saved as ipl_score_predictor.pkl")

# Get available options for each feature
def get_unique_options(column):
    return sorted(df[column].dropna().unique())

venues = get_unique_options('venue')
teams = get_unique_options('team1')  # team1 and team2 have the same set
cities = get_unique_options('city')
seasons = get_unique_options('season')
toss_decisions = get_unique_options('toss_decision')

# Prediction function
def predict_score(input_data):
    try:
        model = joblib.load('ipl_score_predictor.pkl')
        df_input = pd.DataFrame([input_data], columns=features)
        prediction = model.predict(df_input)[0]
        return round(prediction)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None

# User input section
def get_user_input():
    print("Please select from the following options:")

    print("\nVenues:")
    for idx, v in enumerate(venues):
        print(f"{idx+1}. {v}")
    venue = venues[int(input("Select venue (number): ")) - 1]

    print("\nTeams:")
    for idx, t in enumerate(teams):
        print(f"{idx+1}. {t}")
    team1 = teams[int(input("Select team1 (number): ")) - 1]
    team2 = teams[int(input("Select team2 (number): ")) - 1]

    print("\nToss Winner:")
    for idx, t in enumerate(teams):
        print(f"{idx+1}. {t}")
    toss_winner = teams[int(input("Select toss winner (number): ")) - 1]

    print("\nToss Decision:")
    for idx, td in enumerate(toss_decisions):
        print(f"{idx+1}. {td}")
    toss_decision = toss_decisions[int(input("Select toss decision (number): ")) - 1]

    print("\nCities:")
    for idx, c in enumerate(cities):
        print(f"{idx+1}. {c}")
    city = cities[int(input("Select city (number): ")) - 1]

    print("\nSeasons:")
    for idx, s in enumerate(seasons):
        print(f"{idx+1}. {s}")
    season = seasons[int(input("Select season (number): ")) - 1]

    target_overs = float(input("\nEnter target overs (e.g., 20): "))

    return {
        'venue': venue,
        'team1': team1,
        'team2': team2,
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'city': city,
        'season': season,
        'target_overs': target_overs
    }

if __name__ == "__main__":
    print("\n--- IPL Target Score Predictor ---\n")
    user_input = get_user_input()
    prediction = predict_score(user_input)
    if prediction:
        print(f"\nPredicted Target Score for Chasing Team: {prediction}")
    else:
        print("Prediction failed.")
