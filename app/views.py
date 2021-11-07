from app import application
from flask import render_template, request


from app.models import model_predict as mp


@application.route('/predictByUrl<path:incomingURL>', methods=['GET', 'POST'])
def hello_world():  # put application's code here
    data = request.form
    # html_text = requests.get(data['analyseURL']).text
    values = mp.predict_by_url(data['analyseURL'])
    return render_template("привет мир")


@application.route('/')
def hell_world():  # put application's code here
    return render_template("привет мир")


