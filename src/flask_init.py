
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
    # print(type(results[0]))
    print(type(results[0]))
    # for item in results:
    #     print
    # return render_template("home.html")
    return render_template("classification_result.html", text = text, classification = results)
    
@app.route('/trained', methods=['GET', 'POST'])
def home_train():
    if request.form["button"]=='default':
        
        path = os.getcwd() + "/data/corpus"
        # X, y = load_corpus(path)
        session["estimators_corpus"] = path



        
        #Train model
    elif request.form["button"]=='load':
        path = request.form["path"]
        # X, y = load_corpus(path)
        session["estimators_corpus"] = path
        #TRain model
        pass
    elif request.form["button"]=='craw':
        return render_template("craw.html")
        
    return render_template("home.html")

#     engine = TextEngine()
#     cwdpath = os.getcwd()
#     path = os.path.join(cwdpath,"corpus")
#     query = os.path.join(cwdpath,"queries\\all.txt")
#     with open(query) as json_file:
#         data = json.load(json_file)

#     pdf,text = explore_directories(path)
    
#     for txt_path in text:
#         p = txt_to_text(txt_path)
#         engine.insert_document(txt_path,p)
#     count = 0
    
#     prec_lsi = []
#     recall_lsi = []
#     f_lsi = []
#     f1_lsi = []
#     rm_lsi = []

#     prec_lsi_expanded = []
#     recall_lsi_expanded = []
#     f_lsi_expanded = []
#     f1_lsi_expanded = []
#     rm_lsi_expanded = []

#     prec_vect = []
#     recall_vect = []
#     f_vect = []
#     f1_vect = []
#     rm_vect = []

#     for elem in data:
#         if count == 30:
#             break
#         q = elem["Text"]
#         relevant = elem["RelevantDocuments"]
        
#         r_lsi = engine.query_LSI(10,q)
#         r_lsi_expanded = engine.query_LSI(10,q,True)
#         r_vect = engine.query_tf_idf(q)
        
#         epsilon_lsi = 0.5
#         quantity_vect = 20
        
#         temp_vect = r_vect[:quantity_vect]
#         temp_lsi = list(filter(lambda x: x[1]>epsilon_lsi,r_lsi))
#         temp_lsi_expanded = list(filter(lambda x: x[1]>epsilon_lsi,r_lsi_expanded))
        
#         rec_vect = [a[0] for a in temp_vect ]
#         rec_lsi = [a[0] for a in temp_lsi ]
#         rec_lsi_expanded = [a[0] for a in temp_lsi_expanded ]
        
#         recovered_vect = []
#         recovered_lsi = []
#         recovered_lsi_expanded = []
        
#         for elem in  rec_vect:
#             recovered_vect.append(elem.rsplit("\\",1)[-1])
#         for elem in  rec_lsi:
#             recovered_lsi.append(elem.rsplit("\\",1)[-1])
#         for elem in  rec_lsi_expanded:
#             recovered_lsi_expanded.append(elem.rsplit("\\",1)[-1])

#         prec_vect.append(precision_score(recovered_vect,relevant)) 
#         recall_vect.append(recall(recovered_vect,relevant))
#         f_vect.append(f_metric(1,recovered_vect,relevant))
#         f1_vect.append(f1_metric(recovered_vect,relevant))
#         rm_vect.append(r_precission(5,recovered_vect,relevant))

#         prec_lsi.append(precision_score(recovered_lsi,relevant)) 
#         recall_lsi.append(recall(recovered_lsi,relevant))
#         f_lsi.append(f_metric(1,recovered_lsi,relevant))
#         f1_lsi.append(f1_metric(recovered_lsi,relevant))
#         rm_lsi.append(r_precission(5,recovered_lsi,relevant))
        
#         prec_lsi_expanded.append(precision_score(recovered_lsi_expanded,relevant)) 
#         recall_lsi_expanded.append(recall(recovered_lsi_expanded,relevant))
#         f_lsi_expanded.append(f_metric(1,recovered_lsi_expanded,relevant))
#         f1_lsi_expanded.append(f1_metric(recovered_lsi_expanded,relevant))
#         rm_lsi_expanded.append(r_precission(5,recovered_lsi_expanded,relevant))
        
        
#         count+=1
    
#     prec_lsi_expanded = sum(prec_lsi_expanded)/len(prec_lsi_expanded)
#     recall_lsi_expanded = sum(recall_lsi_expanded)/len(recall_lsi_expanded)
#     f_lsi_expanded = sum(f_lsi_expanded)/len(f_lsi_expanded)
#     f1_lsi_expanded = sum(f1_lsi_expanded)/len(f1_lsi_expanded)
#     rm_lsi_expanded = sum(rm_lsi_expanded)/len(rm_lsi_expanded)
    
#     prec_lsi = sum(prec_lsi)/len(prec_lsi)
#     recall_lsi = sum(recall_lsi)/len(recall_lsi)
#     f_lsi = sum(f_lsi)/len(f_lsi)
#     f1_lsi = sum(f1_lsi)/len(f1_lsi)
#     rm_lsi = sum(rm_lsi)/len(rm_lsi)

#     prec_vect = sum(prec_vect)/len(prec_vect)
#     recall_vect = sum(recall_vect)/len(recall_vect)
#     f_vect = sum(f_vect)/len(f_vect)
#     f1_vect = sum(f1_vect)/len(f1_vect)
#     rm_vect = sum(rm_vect)/len(rm_vect)
    
#     return render_template("metrics_index.html",prec_lsi_expanded = prec_lsi_expanded, rec_lsi_expanded = recall_lsi_expanded, f_lsi_expanded = f_lsi_expanded, f1_lsi_expanded = f1_lsi_expanded, r_lsi_expanded = rm_lsi_expanded,prec_lsi = prec_lsi, rec_lsi = recall_lsi, f_lsi = f_lsi, f1_lsi = f1_lsi, r_lsi = rm_lsi, prec_vect = prec_vect, rec_vect = recall_vect, f_vect = f_vect, f1_vect = f1_vect, r_vect = rm_vect )#,p = precission, rec = rec, f = f, f1 = f1, r = r_precission )

# @app.route('/result', methods=['GET', 'POST'])
# def result():
#     if app.engine is None:
#         app.engine = TextEngine()
#     model = request.form['model']

#     path = request.form['path']
#     query = request.form['query']

#     if path != session['path']:
#         query = request.form['query']
#         pdfs,text = explore_directories(path)
#         documents = []
#         for pdf_path in pdfs:
#             p = pdf_to_text(pdf_path)
#             name = pdf_path.split('\\')[-1]
#             documents.append((pdf_path,pdf_path))

#             app.engine.insert_document(pdf_path,p)
#         for txt_path in text:
#             p = txt_to_text(txt_path)
#             name = txt_path.split('\\')[-1]
#             documents.append((txt_path,name))
#             app.engine.insert_document(txt_path,p)
#         session['path'] = path
#         session['query'] = query
#     r = None
#     if model == 'vectorial':
#         r = app.engine.query_tf_idf(query)
#         return render_template("result.html",documents = r)
#     else:
#         k = 10
#         r = app.engine.query_LSI(k,query)
#         rec = extras.recom(app.engine.tf_idf_matrix, app.engine.documents)
#         return render_template("result_r.html",documents = r, rec = rec)
    
# @app.route('/result/<path:f>', methods=['GET', 'POST'])
# def pdf(f):
#     spl = f.rsplit("/",1)
#     n = spl.pop(-1)
#     path = spl[0]
#     return send_from_directory(path,n)

if __name__ == '__main__':
    app.run()
