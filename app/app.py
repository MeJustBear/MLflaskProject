from flask import Flask, request, jsonify
import config
# from app.models.predict import model_predict as mp


application = Flask(__name__)
application.config.from_object(config.DevelopementConfig)


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

if __name__ == '__main__':
    application.run(host=config.ApplicationConfig.host, port=config.ApplicationConfig.port)
