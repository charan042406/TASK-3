# CODTECH Internship – Task 3

This project is an end-to-end data science workflow from data collection to deployment using Flask.

## Steps:
- Data Collection: Titanic dataset
- Preprocessing: Handling missing values and feature selection
- Modeling: Logistic Regression
- Deployment: Flask API

## Run the project:

1. Train the model:
```bash
python train_model.py
```

2. Run the Flask app:
```bash
python app.py
```

3. Make predictions using POST request to `/predict` with JSON:
```json
{ "features": [3, 22, 7.25] }
```
