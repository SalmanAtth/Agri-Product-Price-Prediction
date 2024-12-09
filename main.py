from flask import Flask, request, jsonify, render_template_string
import numpy as np
import pickle

# Load your trained machine learning model
# Assuming you have saved your model as 'your_model.pkl'
model = pickle.load(open('lr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)
# Start ngrok when app is run

# HTML Template (since you don't have a template directory in Jupyter Notebook)
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Commodity Price Prediction</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-9ndCyUaIbzAi2FUVXJi0CjmCapSmO7SnpJef0486qhLnuZ2cdeRhO02iuK6FUUVM" crossorigin="anonymous">

</head>
<body>
    <h1 class="text-center text-success">Crop Price Prediction</h1>

    <div class="container my-4 mt-4" style="background-color: rgba(0, 0, 0, 0.5); border-radius: 20px; color:white">

        <h1 class="text-center text-danger">Predict Agriculture Commodity Price</h1>
        <form method="POST" action="/predict">
            <label for="State">State:</label>
            <input type="text" class="form-control" name="State" step="any"  required>

            <label for="District">District:</label>
            <input type="text" class="form-control" name="District" step="any" required>
            <label for="Market">Market:</label>
            <input type="text" class="form-control" name="Market" step="any"  required>

            <label for="Commodity">Commodity:</label>
            <input type="text" class="form-control" name="Commodity" step="any"  required>

            <label for="Variety">Variety:</label>
            <input type="text" class="form-control" name="Variety" step="any"  required>

            <label for="month_column">Month:</label>
            <input type="text" class="form-control" name="month_column" step="any"  required>

            <label for="season_names">Season:</label>
            <input type="text" class="form-control" name="season_names" step="any" required>

            <label for="day">Day:</label>
            <input type="number" class="form-control" name="day" step="any" required>

            <button type="submit" class="btn btn-danger btn-lg mt-2 btn-block">Predict</button>


        </form>
        <div class="result">
     {% if prediction %}
       <h1 class="text-center"> Predicted Price: <br>{{prediction}}</h1>

      {% endif %}
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js" integrity="sha384-geWF76RCwLtnZ8qwWowPQNguL3RmwHVBC9FhGdlKrxdiJJigb/j/68SIy3Te4Bkz" crossorigin="anonymous"></script>

</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(html_template)


@app.route('/predict', methods=['POST'])
def predict():
    print("hello")
    if request.method == 'POST':
        # Retrieve form data
        State = request.form['State']
        District = request.form['District']
        Market = request.form['Market']
        Commodity = request.form['Commodity']
        Variety = request.form['Variety']

        month_column = request.form['month_column']

        season_names = request.form['season_names']
        day = int(request.form['day'])

        # Arrange data into a format suitable for the model
        input_features = np.array(
            [[State, District, Market, Commodity, Variety, month_column, season_names, day]], dtype=object)
        #     State,District,Market,Commodity,Variety,month_column,season_names,day
        transformed_features = preprocessor.transform(input_features)
        # Make the prediction
        prediction = model.predict(transformed_features).reshape(1, -1)

        # Render the result on the same page
        return render_template_string(html_template, prediction=f'Predicted Price: {prediction[0]}')


# Run the Flask app
app.run(debug=True)