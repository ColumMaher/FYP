from flask import Flask, request, render_template
import json
import main

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_end_predictions', methods=['post'])
def get_prediction_eos():
    try:
        input_text = ' '.join(request.json['input_text'].split())
        top_k = request.json['top_k']
        res = main.get_model_predictions(input_text, top_clean=int(top_k))
        return app.response_class(response=json.dumps(res), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')

if __name__ == "__main__":
    app.run()
