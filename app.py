import os
from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity
import io
import base64
import csv

app = Flask(__name__)

# Carregar o modelo MobileNetV2 pré-treinado
print("Carregando modelo MobileNetV2...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("Modelo MobileNetV2 carregado com sucesso.")

# Carregar características dos pets a partir do arquivo CSV
pet_features = np.loadtxt('pet_features.csv', delimiter=',')

# Carregar informações dos arquivos dos pets a partir do arquivo CSV
with open('pet_files.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Ignorar o cabeçalho
    pet_files = [row[0] for row in reader]

print("Características e informações dos arquivos dos pets carregadas com sucesso.")


# Função para encontrar o pet mais semelhante
def find_most_similar_pet(query_features):
    similarities = cosine_similarity([query_features], pet_features)
    most_similar_indices = similarities.argsort()[0][-5:][::-1]
    most_similar_files = [pet_files[i] for i in most_similar_indices]
    images_base64 = []
    for file in most_similar_files:
        with open(os.path.join('imagens', file), 'rb') as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            images_base64.append(image_base64)
    return most_similar_files, images_base64


# Rota de predição
@app.route('/predict', methods=['POST'])
def predict():
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
        most_similar_pets, images_base64 = find_most_similar_pet(query_features)
        return jsonify({'most_similar_pets': most_similar_pets, 'images_base64': images_base64}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
