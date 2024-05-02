import flask
from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
import subprocess


app = flask.Flask(__name__)
app.config["DEBUG"] = True

path_base = "/home/dsonlineli/Taller_Despliegue/"

colaboradores = [
    {"colab_id": 1, "name": "Alba", "city": "Barcelona", "age": 28},
    {"colab_id": 2, "name": "Enrique", "city": "Madrid", "age": 28},
    {"colab_id": 3, "name": "Lucas", "city": "Sevilla", "age": 26},
    {"colab_id": 4, "name": "Martín", "city": "Valencia", "age": 31}]

####################
@app.route('/', methods=['GET'])
def home():
    return "<h1> Bienvenido al prototipo de API de 'LA PINTA'</h2><p> Esta API analiza el indice (contar un poco la historia...)<p> Para realizar una predicción escribe: --> 'http://127.0.0.1:5000/api/v1/predict'.<p> Para consultar los colaboradores: --> 'http://127.0.0.1:5000/api/v1/colaboradores/all'</p>"

####################
@app.route('/api/v1/colab', methods=['GET'])
def colab():
    return "Esta API ha sido desarrollada por Alba, Enrique, Lucas y Martín"

###################
@app.route('/api/v1/colaboradores/all', methods=['GET'])
def get_colaboradores():

    nombres = [colaborador["name"] for colaborador in colaboradores]

    return jsonify({'colaboradores': nombres})

###################
@app.route('/api/v1/colaboradores', methods=['GET'])
def colab_id():
    if 'colab_id' in request.args:  
        id = int(request.args['colab_id'])
    else:
        return "Error: El colab_id no es válido. Por favor, prueba con uno de los siguientes valores: [1, 2, 3, 4]."
    
    results = []
    for colaborador in colaboradores:
        if colaborador['colab_id'] == id:
            results.append(colaborador)
    return jsonify(results)


###################

###################

@app.route('/api/v1/predict', methods=['GET'])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET

    model = pickle.load(open(path_base + 'ad_model.pkl','rb'))
    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    print(tv,radio,newspaper)
    print(type(tv))

    if tv is None or radio is None or newspaper is None:
        return "Args empty, the data are not enough to predict, STUPID!!!!"
    else:
        prediction = model.predict([[float(tv),float(radio),float(newspaper)]])
    
    return jsonify({'predictions': prediction[0]})

###################

@app.route('/webhook_2024', methods=['POST'])
def webhook():
    # Ruta al repositorio donde se realizará el pull
    path_repo = '/ruta/a/tu/repositorio/en/PythonAnywhere'
    servidor_web = '/ruta/al/fichero/WSGI/de/configuracion' 

    # Comprueba si la solicitud POST contiene datos JSON
    if request.is_json:
        payload = request.json
        # Verifica si la carga útil (payload) contiene información sobre el repositorio
        if 'repository' in payload:
            # Extrae el nombre del repositorio y la URL de clonación
            repo_name = payload['repository']['name']
            clone_url = payload['repository']['clone_url']
            
            # Cambia al directorio del repositorio
            try:
                os.chdir(path_repo)
            except FileNotFoundError:
                return jsonify({'message': 'El directorio del repositorio no existe'}), 404

            # Realiza un git pull en el repositorio
            try:
                subprocess.run(['git', 'pull', clone_url], check=True)
                subprocess.run(['touch', servidor_web], check=True) # Trick to automatically reload PythonAnywhere WebServer
                return jsonify({'message': f'Se realizó un git pull en el repositorio {repo_name}'}), 200
            except subprocess.CalledProcessError:
                return jsonify({'message': f'Error al realizar git pull en el repositorio {repo_name}'}), 500
        else:
            return jsonify({'message': 'No se encontró información sobre el repositorio en la carga útil (payload)'}), 400
    else:
        return jsonify({'message': 'La solicitud no contiene datos JSON'}), 400

if __name__ == '__main__':
    app.run()
