import uuid


class PathsConfig:
    path_to_train_data = "static\\train\\"
    url_train_data = "https://github.com/ods-ai-ml4sg/proj_news_viz/releases/download/data/rt.csv.gz"
    path_to_model = "static\\predict\\"
    model_name = "model"
    encoder_name = "encoder_classes.npy"
    tokenizer_name = "tokenizer.json"


class ApplicationConfig:
    host = 'localhost'
    port = 8000
    appName = 'predictor'


class BaseConfig:
    SECRET_KEY = uuid.uuid4().hex
    FLASK_APP = ApplicationConfig.appName
    host = 'localhost'
    port = 8000
    # SERVER_NAME = 'server.dev'


class DevelopementConfig(BaseConfig):
    DEBUG = True


class ProductionConfig(BaseConfig):
    DEBUG = False


class forJSONexample:
    data = {"Без политики": "50.7047 %", "Бывший СССР": "0.0 %", "Мероприятия RT": "0.0 %",
            "Мир": "0.0 %", "Наука": "49.2952 %", "Новости партнёров": "0.0 %", "Пресс - релизы": "0.0 %",
            "Россия": "0.0 %", "Спорт": "0.0 %", "Экономика": "0.0 %"}
