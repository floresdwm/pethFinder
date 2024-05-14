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

app = Flask(__name__)
swagger = Swagger(app)

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
    most_similar_indices = similarities.argsort()[0][-20:][::-1]
    most_similar_files = [pet_files[i] for i in most_similar_indices]

    # Formatar os nomes dos arquivos
    formatted_most_similar_files = []
    for file in most_similar_files:
        # Extrair o número do ID do arquivo
        id_number = file.split('_id_')[1].split('_')[0]
        # Formatar a URL com base no número do ID
        url = f"https://petsrs.com.br/pet/{id_number}"
        formatted_most_similar_files.append(url)

    return formatted_most_similar_files


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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
