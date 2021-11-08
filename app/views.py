from app import application
from flask import request

from app.models.predict import model_predict as mp


@application.route('/predictByUrl', methods=['GET', 'POST'])
def hello_world():  # put application's code here
    data = request.form
    # html_text = requests.get(data['analyseURL']).text
    values = mp.predict_by_url(data['analyseURL'])
    return "привет мир"


@application.route('/')
def hell_world():  # put application's code here
    return "привет мир"


