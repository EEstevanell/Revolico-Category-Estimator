
from flask import Flask,render_template,request,make_response, session, send_from_directory
from main import *
from crawler.main_sp import *
from crawler.main_sp import *
import os
from multiprocessing import Process
import json
app = Flask(__name__)
SESSION_TYPE = 'filesystem'
estimators = None
app.secret_key = 'some secret key' 
app.engine = None

@app.route('/', methods=['GET', 'POST'])
def index():
    session['path'] = ''
    session["estimators_corpus"] = None
    session["estimators"] = False
    return render_template("home.html")

@app.route('/home_decide', methods=['GET', 'POST'])
def home_decide():
    if request.form["button"]=='train':
        return render_template("home_train.html")
    else:
        return render_template("select_text.html")
@app.route('/craw', methods=['GET', 'POST'])
def craw():
    global estimators
    seed = request.form["seed"]
    limit = int(request.form["limit"])
    p1 = Process(target = run_crawler, args = (limit,seed) )
    p1.start()
    p1.join()
    # run_crawler(limit,seed)
    path = os.getcwd() +'/crawler/websites'
    x,y = get_raw_data(path)
    path = os.getcwd() + "/data/corpus"
    save_corpus(path,x,y)
    session["estimators_corpus"] = path

    try:
        X, y = load_corpus(path)
    except:
        return render_template("error_craw.html")
    estimators = fit(X, y)
    session["estimators"] = True
    return render_template("home.html")

@app.route('/select_text', methods=['GET', 'POST'])
def select_text():
    global estimators
    if session["estimators_corpus"] is None:
        path = os.getcwd() + "/data/corpus"
        
        session["estimators_corpus"] = path

    if request.form["button"] == 'path':
        path = request.form["path"]
        text = read_text(path)
    else:
        text = [request.form["single_text"]]
    path = session["estimators_corpus"]
    
    if not session["estimators"]:
        X, y = load_corpus(path)
        estimators = fit(X, y)
        session["estimators"] = True

    result = predict(text, estimators)
    results = [a.tolist() for a in result]
    return render_template("classification_result.html", text = text, classification = results)
    
@app.route('/trained', methods=['GET', 'POST'])
def home_train():
    global estimators
    if request.form["button"]=='default':
        path = os.getcwd() + R"\data\corpus"
        session["estimators_corpus"] = path

    elif request.form["button"]=='load':
        path = request.form["path"]
        session["estimators_corpus"] = path
        
    elif request.form["button"]=='craw':
        return render_template("craw.html")
    
    print("training from path:", path)
    try:
        X, y = load_corpus(path)
    except:
        return render_template("error_home_train.html")

    estimators = fit(X, y)
    session["estimators"] = True
    return render_template("home.html")


if __name__ == '__main__':
    app.run()
