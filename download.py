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

import subprocess

# Executa o comando 'pip freeze' para listar os pacotes instalados
pip_freeze_output = subprocess.check_output(['pip', 'freeze']).decode()

# Escreve a saída do 'pip freeze' em um arquivo 'requirements.txt'
with open('requirements2.txt', 'w') as requirements_file:
    requirements_file.write(pip_freeze_output)
