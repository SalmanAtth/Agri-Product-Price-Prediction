from flask import Flask, request, render_template
import numpy as np
import pickle

# Load your trained machine learning model
# Assuming you have saved your model as 'your_model.pkl'
model = pickle.load(open('lr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
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
        return render_template('index.html', prediction=f'Predicted Price: {prediction[0]}')


# Run the Flask app
app.run(debug=True)
