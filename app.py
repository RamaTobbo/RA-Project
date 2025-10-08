from flask import Flask, render_template, jsonify
from flask_cors import CORS
from Code import Michalewicz_Code # import your new module

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('div.html')  # make sure div.html is in /templates

@app.route('/vis')
def vis():
    return render_template('vis.html')
@app.route('/learn')
def learn():
    return render_template('learn.html')
@app.route('/summary_act')
def summary_act():
    return render_template('summary_act.html')
@app.route('/bat_act')
def bat_act():
    return render_template('bat_act.html')
@app.route('/wolf_act')
def wolf_act():
    return render_template('wolf_act.html')
@app.route('/bee_act')
def bee_act():
    return render_template('bee_act.html')
@app.route('/avg_sum_ackley')
def avg_sum_ackley():
    return render_template('avg_sum_ackley.html')
@app.route('/best_sum_ackley')
def best_sum_ackley():
    return render_template('best_sum_ackley.html')
@app.route('/wolf_ackley')
def wolf_ackley():
    return render_template('wolf_ackley.html')
@app.route('/wolf_michalewicz')
def wolf_michalewicz():
    return render_template('wolf_michalewicz.html')
@app.route('/bee_ackley')
def bee_ackley():
    return render_template('bee_ackley.html')
@app.route('/bee_michalewicz')
def bee_michalewicz():
    return render_template('bee_michalewicz.html')
@app.route('/wolf')
def wolf():
    return render_template('Wolf/index.html')

@app.route('/bee')
def bee():
    return render_template('Bee/index.html')

@app.route('/bat')
def bat():
    return render_template('Bat/index.html')

@app.route('/fish')
def fish():
    return render_template('Fish/index.html')

@app.route('/run', methods=['POST'])

def run_algorithm():
    result = Michalewicz_Code.run_algorithm()  # call the function
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
