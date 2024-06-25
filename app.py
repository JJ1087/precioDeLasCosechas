from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el escalador
model = joblib.load('randomForest.pkl')
scaler = joblib.load('scaler.pkl')
app.logger.debug('Modelo y escalador cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        CostCultivation = float(request.form['CostCultivation'])
        CostCultivation2 = float(request.form['CostCultivation2'])
        Yield = float(request.form['Yield'])
        RainFallAnnual = float(request.form['RainFallAnnual'])

        # Crear un DataFrame con todas las características necesarias
        data = {
            'State': [0],  
            'Crop': [0],  
            'CostCultivation': [CostCultivation],
            'CostCultivation2': [CostCultivation2], 
            'Production': [0],
            'Yield': [Yield],  
            'Temperature': [0],
            'RainFall Annual': [RainFallAnnual]
        }

        input_data = pd.DataFrame(data)
        app.logger.debug(f'DataFrame de entrada creado: {input_data}')

        # Escalar los datos de entrada
        scaled_data = scaler.transform(input_data)

        # Seleccionar solo las características usadas para el modelo
        scaled_data_for_prediction = scaled_data[:, [2, 3, 5, 7]]  # Asegúrate de que estos índices son correctos

        # Realizar la predicción con los datos escalados
        prediccion = model.predict(scaled_data_for_prediction)
        app.logger.debug(f'Predicción: {prediccion[0]}')

        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediccion[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
