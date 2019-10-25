# encoding:utf-8
"""
使用flask部署深度学习模型
采用REST API
默认就是在cuda上操作
"""
import flask
from predict import Chatbot
import json
app = flask.Flask(__name__)


def load_model():
    """
    Load the pre-trained model
    :return: 
    """
    global model
    config = json.load(open("chatbot.json", "r"))
    model = Chatbot(config)


def prepare_seq(input_seq):

    result = model.evalModel(input_seq)

    return result

@app.route("/predict/", methods=["POST"])
def predict():

    data = {"success": False}
    if flask.request.method == "POST":

        message = flask.request.form["message"]
        result = prepare_seq(message)
        print(message, result)
        data["result"] = result
        data["success"] = True
    return flask.jsonify(data)

def main():
    load_model()
    app.run()

if __name__ == '__main__':
    main()


