import os
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Carregar o modelo MobileNetV2 pré-treinado
print("Carregando modelo MobileNetV2...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
print("Modelo MobileNetV2 carregado com sucesso.")

# Carregar imagens de pets e extrair características
print("Carregando imagens de pets e extraindo características...")
pet_features = []
pet_files = []
for file in os.listdir('imagens'):
    if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
        file_path = os.path.join('imagens', file)
        pet_files.append(file)
        img = Image.open(file_path).resize((224, 224))
        img = np.array(img)
        img = preprocess_input(img)
        features = base_model.predict(np.expand_dims(img, axis=0))
        features = features.flatten()
        pet_features.append(features)
pet_features = np.array(pet_features)
print("Características extraídas com sucesso.")


# Função para encontrar o pet mais semelhante
def find_most_similar_pet(query_features):
    similarities = cosine_similarity([query_features], pet_features)
    most_similar_indices = similarities.argsort()[0][-5:][::-1]
    return [pet_files[i] for i in most_similar_indices]


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
        most_similar_pets = find_most_similar_pet(query_features)
        return jsonify({'most_similar_pets': most_similar_pets}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
