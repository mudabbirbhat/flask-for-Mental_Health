from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the model and label encoder
with open('model_RandomForest.pkl', 'rb') as f:
    model, label_encoder = pickle.load(f)

# Define a root route
@app.route('/')
def home():
    return "Welcome to the Disorder Prediction API! Use the /predict endpoint to make predictions."

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Extracting features from form data
    features = [
        request.form.get('feeling.nervous', type=float),
        request.form.get('panic', type=float),
        request.form.get('breathing.rapidly', type=float),
        request.form.get('sweating', type=float),
        request.form.get('trouble.in.concentration', type=float),
        request.form.get('having.trouble.in.sleeping', type=float),
        request.form.get('having.trouble.with.work', type=float),
        request.form.get('hopelessness', type=float),
        request.form.get('anger', type=float),
        request.form.get('over.react', type=float),
        request.form.get('change.in.eating', type=float),
        request.form.get('suicidal.thought', type=float),
        request.form.get('feeling.tired', type=float),
        request.form.get('close.friend', type=float),
        request.form.get('social.media.addiction', type=float),
        request.form.get('weight.gain', type=float),
        request.form.get('material.possessions', type=float),
        request.form.get('introvert', type=float),
        request.form.get('popping.up.stressful.memory', type=float),
        request.form.get('having.nightmares', type=float),
        request.form.get('avoids.people.or.activities', type=float),
        request.form.get('feeling.negative', type=float),
        request.form.get('trouble.concentrating', type=float),
        request.form.get('blamming.yourself', type=float),
    ]

    # Convert to NumPy array for prediction
    features_array = np.array([features])

    # Make prediction
    prediction = model.predict(features_array)

    # Convert numerical prediction to the disorder label
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    # Return the result as a JSON response
    return jsonify({
        'prediction': predicted_label
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
