import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import csv
from flasgger import Swagger
from flask import render_template
import chardet
import cv2
import torch

app = Flask(__name__, static_url_path='/static')
swagger = Swagger(app)

# Função para carregar o modelo YOLOv5
def load_yolo():
    # Carregar modelo YOLOv5
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.eval()
    return model


def detect_objects(model, img):
    results = model(img)

    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        if int(cls) == 16 and conf > 0.4:  # 16 é a classe de animais
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
            # Salvar a imagem recortada p test
            #cv2.imwrite('cropped_temp.jpg', cropped_img)
            return cropped_img
        if int(cls) == 15 and conf > 0.4:  # 15 é a classe de animais
            cropped_img = img[int(y1):int(y2), int(x1):int(x2)]
            # Salvar a imagem recortada p test
            #cv2.imwrite('cropped_temp2.jpg', cropped_img)
            return cropped_img

    return None


# Carregar YOLOv5
print("Carregando modelo yolov5...")
net = load_yolo()
print("Carregado modelo yolov5!")

# Carregar o modelo MobileNetV2 pré-treinado
print("Carregando modelo MobileNetV2...")
model_path = "MobileNetV2_model.h5"
base_model = MobileNetV2(weights=model_path, include_top=False, input_shape=(224, 224, 3))
print("Modelo MobileNetV2 carregado com sucesso.")

# Função para determinar a codificação do arquivo CSV
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
        encoding = result['encoding']
    return encoding

# Carregar características dos pets a partir do arquivo CSV
pet_features = np.loadtxt('pet_features.csv', delimiter=',')
print("pet_features carregadas com sucesso.")

# Carregar informações dos arquivos dos pets a partir do arquivo CSV
pet_files_path = 'pet_files.csv'
csv_encoding = detect_encoding(pet_files_path)
with open(pet_files_path, 'r', encoding=csv_encoding) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Ignorar o cabeçalho
    pet_files = [row[0] for row in reader]

print("pet_files carregadas com sucesso.")


# Função para encontrar o pet mais semelhante
def find_most_similar_pet(query_features):
    similarities = cosine_similarity([query_features], pet_features)
    most_similar_indices = similarities.argsort()[0][-20:][::-1]
    most_similar_files = [pet_files[i] for i in most_similar_indices]

    # Formatar os nomes dos arquivos
    formatted_most_similar_files = []
    for file in most_similar_files:
        # Extrair o número do ID do arquivo
        #id_number = file.split('_id_')[1].split('_')[0]
        id_number = file.split('.')[0]
        # Formatar a URL com base no número do ID
        url = f"https://petsrs.com.br/pet/{id_number}"
        formatted_most_similar_files.append(url)

    return formatted_most_similar_files


@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('home.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('home.html', error='No selected file')
        if file:
            # Salvar a imagem temporariamente
            image_path = 'temp.jpg'
            file.save(image_path)

            # Carregar imagem
            image = cv2.imread(image_path)

            # Detectar e recortar objetos na imagem yolo
            cropped_image = detect_objects(net, image) 

            if cropped_image is not None:
                # Processar yolo
                img = Image.fromarray(cropped_image).resize((224, 224))
                img = np.array(img)
                img = preprocess_input(img)
                query_features = base_model.predict(np.expand_dims(img, axis=0)).flatten()
                most_similar_pets = find_most_similar_pet(query_features)
                
                # Remover imagem temporária
                os.remove(image_path)

                return render_template('results.html', most_similar_pets=most_similar_pets)
            else:
                # Remover imagem temporária
                os.remove(image_path)

                img = Image.open(file).resize((224, 224))
                img = np.array(img)
                img = preprocess_input(img)
                query_features = base_model.predict(np.expand_dims(img, axis=0)).flatten()
                most_similar_pets = find_most_similar_pet(query_features)            
                return render_template('results.html', most_similar_pets=most_similar_pets)
    return render_template('home.html')


@app.route('/', methods=['GET', 'POST'])
def idx():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('home.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('home.html', error='No selected file')
        if file:
            # Salvar a imagem temporariamente
            image_path = 'temp.jpg'
            file.save(image_path)

            # Carregar imagem
            image = cv2.imread(image_path)

            # Detectar e recortar objetos na imagem yolo
            cropped_image = detect_objects(net, image) 

            if cropped_image is not None:
                # Processar yolo
                img = Image.fromarray(cropped_image).resize((224, 224))
                img = np.array(img)
                img = preprocess_input(img)
                query_features = base_model.predict(np.expand_dims(img, axis=0)).flatten()
                most_similar_pets = find_most_similar_pet(query_features)
                
                # Remover imagem temporária
                os.remove(image_path)

                return render_template('results.html', most_similar_pets=most_similar_pets)
            else:
                # Remover imagem temporária
                os.remove(image_path)

                img = Image.open(file).resize((224, 224))
                img = np.array(img)
                img = preprocess_input(img)
                query_features = base_model.predict(np.expand_dims(img, axis=0)).flatten()
                most_similar_pets = find_most_similar_pet(query_features)            
                return render_template('results.html', most_similar_pets=most_similar_pets)
    return render_template('home.html')


# Rota de predição
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para prever os pets mais semelhantes com base em uma imagem enviada.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Imagem do pet (multipart/form-data)
    responses:
      200:
        description: OK
        schema:
          id: Prediction
          properties:
            most_similar_pets:
              type: array
              items:
                type: string
                description: URL para o pet mais semelhante
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        img = Image.open(file).resize((224, 224))
        img = np.array(img)
        img = preprocess_input(img)
        query_features = base_model.predict(np.expand_dims(img, axis=0)).flatten()
        most_similar_pets = find_most_similar_pet(query_features)
        return jsonify({'most_similar_pets': most_similar_pets}), 200
    

# Rota de predição
@app.route('/v2/predict', methods=['POST'])
def predictv2():
    """
    Endpoint para prever os pets mais semelhantes com base em uma imagem enviada.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Imagem do pet (multipart/form-data)
    responses:
      200:
        description: OK
        schema:
          id: Prediction
          properties:
            most_similar_pets:
              type: array
              items:
                type: string
                description: URL para o pet mais semelhante
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        # Salvar a imagem temporariamente
        image_path = 'temp.jpg'
        file.save(image_path)

        # Carregar imagem
        image = cv2.imread(image_path)

        # Detectar e recortar objetos na imagem yolo
        cropped_image = detect_objects(net, image) 

        if cropped_image is not None:
            # Processar yolo
            img = Image.fromarray(cropped_image).resize((224, 224))
            img = np.array(img)
            img = preprocess_input(img)
            query_features = base_model.predict(np.expand_dims(img, axis=0)).flatten()
            most_similar_pets = find_most_similar_pet(query_features)
            
            # Remover imagem temporária
            os.remove(image_path)

            return jsonify({'most_similar_pets': most_similar_pets}), 200
        else:
            # Remover imagem temporária
            os.remove(image_path)

            img = Image.open(file).resize((224, 224))
            img = np.array(img)
            img = preprocess_input(img)
            query_features = base_model.predict(np.expand_dims(img, axis=0)).flatten()
            most_similar_pets = find_most_similar_pet(query_features)
            return jsonify({'most_similar_pets': most_similar_pets}), 200



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
