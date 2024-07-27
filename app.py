from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model using pickle
model_filename = 'real_estate_model.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = request.form.to_dict()
    df = pd.DataFrame([data])

    # Convert data types
    df = df.astype(float)

    # Ensure the input features match the training features
    df = df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'X5 latitude',
             'X6 longitude']]

    # Make prediction
    prediction = model.predict(df)

    # Render the result template with the prediction
    return render_template('result.html', prediction=prediction[0])


if __name__ == '__main__':
    app.run(debug=True)
