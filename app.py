from flask import Flask, render_template, jsonify
from flask_cors import CORS
from Code import Michalewicz_Code # import your new module

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('div.html')  

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
@app.route('/fish_act')
def fish_act():
    return render_template('fish_act.html')
@app.route('/wolf_act')
def wolf_act():
    return render_template('wolf_act.html')
@app.route('/wolf_ack')
def wolf_ack():
    return render_template('Wolf/Wolf Ackley/wolf_ack.html')
@app.route('/fish_ack')
def fish_ack():
    return render_template('Fish/Fish Ackley/fish_ack.html')
@app.route('/bat_ack')
def bat_ack():
    return render_template('Bat/Bat Ackley/bat_ack.html')
@app.route('/bee_ack')
def bee_ack():
    return render_template('Bee/Bee Ackley/bee_ack.html')
@app.route('/bee_act')
def bee_act():
    return render_template('bee_act.html')

@app.route('/best_sum_ackley')
def best_sum_ackley():
    return render_template('best_sum_ackley.html')

@app.route('/wolf_mic')
def wolf_mic():
    return render_template('Wolf/Wolf Michalwicz/wolf_mic.html')


@app.route('/bat_mic')
def bat_mic():
    return render_template('Bat/Bat Michalewicz/bat_mic.html')
@app.route('/fish_mic')
def fish_mic():
    return render_template('Fish/Fish Michalwicz/fish_mic.html')
@app.route('/wolf_ackley')
def wolf_ackley():
    return render_template('Wolf/Wolf Ackley/wolf_ackley.html')
@app.route('/fish_ackley')
def fish_ackley():
    return render_template('Fish/Fish Ackley/fish_ackley.html')

@app.route('/avg_sum_mic')
def avg_sum_mic():
    return render_template('avg_sum_mic.html')
@app.route('/best_sum_mic')
def best_sum_mic():
    return render_template('best_sum_mic.html')

@app.route('/avg_sum_ackley')
def avg_sum_ackley():
    return render_template('avg_sum_ackley.html')
@app.route('/bat_ackley')
def bat_ackley():
    return render_template('Bat/Bat Ackley/bat_ackley.html')
@app.route('/bee_ackley')
def bee_ackley():
    return render_template('Bee/Bee Ackley/bee_ackley.html')
@app.route('/wolf_michalewicz')
def wolf_michalewicz():
    return render_template('Wolf/Wolf Michalwicz/wolf_michalewicz.html')
@app.route('/fish_michalewicz')
def fish_michalewicz():
    return render_template('Fish/Fish Michalwicz/fish_michalewicz.html')
@app.route('/bat_michalewicz')
def bat_michalewicz():
    return render_template('Bat/Bat Michalewicz/bat_michalewicz.html')
@app.route('/bee_michalewicz')
def bee_michalewicz():
    return render_template('Bee/Bee Michalewicz/bee_michalewicz.html')
@app.route('/bee_mic')
def bee_mic():
    return render_template('Bee/Bee Michalewicz/bee_mic.html')
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
@app.route('/fun_fit')
def fun_fit():
    return render_template('fun_fit.html')
@app.route('/mic_fit')
def mic_fit():
    return render_template('mic_fit.html')
@app.route('/ackley_fit')
def ackley_fit():
    return render_template('ackley_fit.html')

# @app.route('/run', methods=['POST'])

# def run_algorithm():
   
#     try:
#         result = Michalewicz_Code.run_algorithm()
#     except Exception as e:
#         result = str(e)
#     return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
