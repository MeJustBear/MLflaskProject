#from app import application
from flask import Flask, render_template, request
import config
from app import models
from app.models import model_predict as mp

app = Flask(__name__)
app.config.from_object(config.ProductionConfig)

@app.route('/predictByUrl', methods=['GET', 'POST'])
def hello_world():  # put application's code here
    data = request.form
    # html_text = requests.get(data['analyseURL']).text
    values = mp.predict_by_url(data['analyseURL'])
    return values


@app.route('/')
def hell_world():  # put application's code here
    return str(mp.predict_by_url(modelE=models.modelE,
                            vgm_url= "https://russian.rt.com/world/article/925240-shoigu-nato-provokaciya-rossiya-chernoe-more",
                             tokenizer=models.tokenizer, encoder=models.encoder))


if __name__=='__main__':
    app.run()

