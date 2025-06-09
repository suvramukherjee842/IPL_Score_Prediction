# IPL Target Score Predictor

A machine learning project to predict the target score for the chasing team in IPL cricket matches, using historical IPL match data.

---

## 📊 Overview

This project trains a Random Forest regression model to predict the target runs a team will need to chase, given match context such as venue, teams, toss, city, season, and overs. The model is trained on real IPL data from `matches.csv`.

---

## 🚀 Features

- Predicts target score for IPL matches using historical data
- Interactive command-line input: select teams, venue, toss, etc.
- Data preprocessing pipeline: handles categorical and numerical features
- Model evaluation with MAE and R² metrics
- Saves trained model for future predictions

---

## 🗂️ Dataset

- **matches.csv**  
  IPL match-level data including teams, scores, venue, winner, toss, and more.  
  *(A sample of the file is included in this repository.)*

---

## 🛠️ Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- joblib
