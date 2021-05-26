from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
nltk.download('stopwords')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data = pd.read_csv(r'D:\Spam Classifier end to end\spam_or_not_spam.csv')
    data['email'] = data['email'].astype(str)
    ps = PorterStemmer()
    corpus = []
    for i in range(0,len(data)) :
        review = re.sub('[^a-zA-Z]', ' ', data['email'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    cv = TfidfVectorizer(max_features=6000)
    x = cv.fit_transform(corpus).toarray()
    y=pd.get_dummies(data['label'])
    y=y.iloc[:,1].values
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.18, random_state = 15)
    spam_detect_model = MultinomialNB().fit(x_train, y_train)
    joblib.dump(spam_detect_model, 'NB_spam_model.pkl')
    NB_spam_model = open('NB_spam_model.pkl','rb')
    spam_detect_model = joblib.load(NB_spam_model)
    
    
    if request.method == 'POST':
        messsage = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = spam_detect_model.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)
        