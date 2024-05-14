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

# Verificar se o modelo MobileNetV2 já está presente no diretório do projeto
model_path = "MobileNetV2_model.h5"
if not os.path.exists(model_path):
    # Baixar o modelo MobileNetV2
    print("Baixando o modelo MobileNetV2...")
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Salvar o modelo baixado
    base_model.save(model_path)
    print("Modelo MobileNetV2 baixado e salvo com sucesso.")
else:
    # Carregar o modelo MobileNetV2 do arquivo
    print("Carregando modelo MobileNetV2...")
    base_model = MobileNetV2(weights=model_path, include_top=False, input_shape=(224, 224, 3))
    print("Modelo MobileNetV2 carregado com sucesso.")