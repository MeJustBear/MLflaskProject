from app import application, config
from flask import jsonify


# from app.models.predict import model_predict as mp


@application.route('/predictByUrl', methods=['GET', 'POST'])
def classify_url():  # put application's code here
    # data = request.form
    # values = mp.predict_by_url(data['analyseURL'])
    # return jsonify(values)
    return jsonify(config.forJSONexample.data)


@application.route('/predictByText', methods=['POST'])
def classify_text():  # put application's code here
    # data = request.form
    # values = mp.predict_by_url(data['analyseURL'])
    # return jsonify(values)
    return jsonify(config.forJSONexample.data)
