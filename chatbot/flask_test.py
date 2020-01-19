# encoding:utf-8

import flask

app = flask.Flask(__name__)

@app.route("/")
def hello_world():
    return "hello world"

if __name__ == "__main__":
    app.run()