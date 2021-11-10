from flask import Flask, request, jsonify
import config

application = Flask(__name__)
application.config.from_object(config.DevelopementConfig)

from .models.predict import model_predict as mp


@application.route('/predictByUrl', methods=['GET', 'POST'])
def classify_url():  # put application's code here
    data = request.form
    values = mp.predict_by_url(data['analyseURL'])
    return jsonify(values)

@application.route('/predictByText', methods=['POST'])
def classify_text():  # put application's code here
    data = request.form
    values = mp.predict_text(data['analyseURL'])
    return jsonify(values)


if __name__ == '__main__':
    application.run(host=config.ApplicationConfig.host, port=config.ApplicationConfig.port)
