import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import joblib
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords



porter = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

def stem_it(text):
    return [porter.stem(word) for word in text]

def stop_it(t):
    dt = [word for word in t if word.lower() not in stop_words]
    return dt

with open('logreg.pkl', 'rb') as file:
    logreg,my_tfidf = joblib.load(file)


from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the request
    user_input = request.json['input']

    # Preprocess the user input
    user_input = word_tokenize(user_input) 
    user_input = stem_it(user_input)
    user_input = stop_it(user_input)
    user_input = ' '.join(user_input)

    # Vectorize the user input
    user_feature = my_tfidf.transform([user_input])

    # Predict the category of the user input using the trained model
    predicted_category = logreg.predict_proba(user_feature)

    # pred=rf.predict(user_feature)

    # Return the predicted category as a JSON response
    predicted_category=predicted_category.tolist()
    ans="True"
    if(predicted_category[0][0]>0.5):
        ans="False"
    response = {'realnews': ans}
    print(predicted_category)
    return jsonify(response)

if __name__ == '__main__':
    app.run()