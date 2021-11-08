from flask import Flask, request, jsonify
import config

application = Flask(__name__)
application.config.from_object(config.DevelopementConfig)


from app.models.predict import model_predict as mp


@application.route('/predictByUrl', methods=['GET', 'POST'])
def classify_url():  # put application's code here
    data = request.form
    values = mp.predict_by_url(data['analyseURL'])
    return jsonify(values)
    # return jsonify(config.forJSONexample.data)


if __name__ == 'main':
    application.run(host=config.ApplicationConfig.host, port=config.ApplicationConfig.port)
