import os
import requests
import numpy as np
import pandas as pd
import pdfplumber
from flask import Flask, render_template, request, jsonify, redirect
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Configuración de la aplicación Flask y la base de datos SQLite
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///estadisticas.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Modelo para las estadísticas de los equipos
class EstadisticasEquipo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre_equipo = db.Column(db.String(50), unique=True, nullable=False)
    ERA = db.Column(db.Float, nullable=False)
    OPS = db.Column(db.Float, nullable=False)
    WHIP = db.Column(db.Float, nullable=False)
    jugadores_clave = db.Column(db.String(200), nullable=False)

# Crear la base de datos y la tabla
with app.app_context():
    db.create_all()

# Función para extraer texto del PDF e inspeccionarlo
def extraer_datos_pdf(ruta_pdf):
    if not os.path.exists(ruta_pdf):
        print(f"Archivo no encontrado: {ruta_pdf}")
        return None
    
    datos = []
    with pdfplumber.open(ruta_pdf) as pdf:
        for pagina in pdf.pages:
            texto = pagina.extract_text()
            if texto:
                lineas = texto.split('\n')
                for linea in lineas:
                    datos.append(linea.split())  # Ajusta según el formato del PDF
    print(datos)  # Para inspeccionar los datos extraídos
    return datos

# Filtrado de las columnas útiles extraídas del PDF
def cargar_datos_pdf(ruta_pdf):
    datos = extraer_datos_pdf(ruta_pdf)
    if datos is None:
        return None

    # Aquí filtramos las columnas de interés si hay demasiadas
    datos_filtrados = []
    for fila in datos:
        if len(fila) >= 4:  # Verificamos que haya al menos 4 columnas
            try:
                era, ops, whip, resultado = fila[0], fila[1], fila[2], fila[3]
                era = float(era)  # Intentar convertir a float
                ops = float(ops)  # Intentar convertir a float
                whip = float(whip)  # Intentar convertir a float
                resultado = int(resultado)  # Intentar convertir a int
                datos_filtrados.append([era, ops, whip, resultado])
            except ValueError:
                continue

    columnas = ['ERA', 'OPS', 'WHIP', 'Resultado']
    df = pd.DataFrame(datos_filtrados, columns=columnas)
    return df

# Entrenamiento del modelo de Machine Learning
def entrenar_modelo(datos):
    if datos is None:
        print("No hay datos para entrenar el modelo.")
        return None

    X = datos[['ERA', 'OPS', 'WHIP']]  # Features
    y = datos['Resultado']  # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    print(f'Precisión del modelo: {accuracy_score(y_test, predicciones)}')
    return modelo

# Cargar los datos del PDF y entrenar el modelo
ruta_pdf = 'ESTADISTICAS.pdf'  # Cambia la ruta si subes el archivo a otro lugar
datos_pdf = cargar_datos_pdf(ruta_pdf)
modelo_ml = entrenar_modelo(datos_pdf) if datos_pdf is not None else None

# Simulación Monte Carlo mejorada
def simulacion_montecarlo(estadisticas_local, estadisticas_visitante, modelo, simulaciones=1000):
    if modelo is None:
        return {"local": 0, "visitante": 0}

    victorias_local = 0
    victorias_visitante = 0

    for _ in range(simulaciones):
        entrada_local = np.array([[estadisticas_local.ERA, estadisticas_local.OPS, estadisticas_local.WHIP]])
        entrada_visitante = np.array([[estadisticas_visitante.ERA, estadisticas_visitante.OPS, estadisticas_visitante.WHIP]])
        
        prob_local = modelo.predict_proba(entrada_local)[0][1]
        prob_visitante = modelo.predict_proba(entrada_visitante)[0][1]
        
        if prob_local > prob_visitante:
            victorias_local += 1
        else:
            victorias_visitante += 1

    return {"local": victorias_local, "visitante": victorias_visitante}

# Función para obtener las estadísticas del equipo
def obtener_estadisticas_equipo(equipo):
    return EstadisticasEquipo.query.filter_by(nombre_equipo=equipo).first()

# Función para obtener los partidos de una fecha específica
def obtener_partidos_por_fecha(fecha):
    url = f"http://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard?dates={fecha}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        partidos = []
        for evento in data["events"]:
            horario_str = evento["date"]
            try:
                horario_formateado = datetime.strptime(horario_str, "%Y-%m-%dT%H:%MZ").strftime("%d de %B, %Y %H:%M")
            except ValueError:
                horario_formateado = horario_str

            competidores = evento["competitions"][0].get("competitors", [])
            if len(competidores) < 2:
                continue

            equipo_local = competidores[0]["team"]["displayName"]
            equipo_visitante = competidores[1]["team"]["displayName"]
            marcador_local = competidores[0].get("score", "N/A")
            marcador_visitante = competidores[1].get("score", "N/A")
            logo_local = competidores[0]["team"].get("logo", "")
            logo_visitante = competidores[1]["team"].get("logo", "")
            outs = evento["competitions"][0].get("situation", {}).get("outs", "N/A")
            bateador_turno = evento["competitions"][0].get("situation", {}).get("currentBatter", {}).get("displayName", "N/A")
            pitcher_turno = evento["competitions"][0].get("situation", {}).get("currentPitcher", {}).get("displayName", "N/A")

            partido = {
                "equipo_local": equipo_local,
                "equipo_visitante": equipo_visitante,
                "horario": horario_formateado,
                "marcador_local": marcador_local,
                "marcador_visitante": marcador_visitante,
                "logo_local": logo_local,
                "logo_visitante": logo_visitante,
                "outs": outs,
                "bateador_turno": bateador_turno,
                "pitcher_turno": pitcher_turno
            }
            partidos.append(partido)
        return partidos
    else:
        return []

# Rutas de Flask
@app.route('/')
def index():
    fecha_actual = datetime.today().strftime('%Y%m%d')
    partidos_hoy = obtener_partidos_por_fecha(fecha_actual)
    return render_template('index.html', partidos=partidos_hoy)

@app.route('/analisis', methods=['GET', 'POST'])
def analisis():
    if request.method == 'POST':
        equipo_local = request.form.get('equipo_local')
        equipo_visitante = request.form.get('equipo_visitante')

        if not equipo_local or not equipo_visitante:
            return render_template('analisis.html', error="Debe ingresar ambos equipos para realizar el análisis.")

        estadisticas_local = obtener_estadisticas_equipo(equipo_local)
        estadisticas_visitante = obtener_estadisticas_equipo(equipo_visitante)

        if estadisticas_local and estadisticas_visitante:
            resultados_simulacion = simulacion_montecarlo(estadisticas_local, estadisticas_visitante, modelo_ml)
            analisis = {
                "equipo_local": equipo_local,
                "equipo_visitante": equipo_visitante,
                "metrics": {
                    "ERA_local": estadisticas_local.ERA,
                    "OPS_local": estadisticas_local.OPS,
                    "WHIP_local": estadisticas_local.WHIP,
                    "ERA_visitante": estadisticas_visitante.ERA,
                    "OPS_visitante": estadisticas_visitante.OPS,
                    "WHIP_visitante": estadisticas_visitante.WHIP,
                },
                "recomendacion_apuesta": f"Victoria esperada para {'local' if resultados_simulacion['local'] > resultados_simulacion['visitante'] else 'visitante'}"
            }
            return render_template('analisis.html', analisis=analisis)
        else:
            return render_template('analisis.html', error="No se encontraron estadísticas para uno o ambos equipos.")
    
    return render_template('analisis.html')

@app.route('/suscripcion')
def suscripcion():
    return render_template('suscripcion.html')

@app.route('/noticias')
def noticias():
    url = "http://site.api.espn.com/apis/site/v2/sports/baseball/mlb/news"
    response = requests.get(url)
    noticias_actuales = response.json().get('articles', [])
    return render_template('noticias.html', noticias=noticias_actuales)

# Rutas para Scoreboard y Calendario
@app.route('/calendario')
def calendario():
    return render_template('calendario.html')

@app.route('/scoreboard')
def scoreboard():
    fecha_actual = datetime.today().strftime('%Y%m%d')
    partidos_hoy = obtener_partidos_por_fecha(fecha_actual)
    return render_template('scoreboard.html', partidos=partidos_hoy)

# API para devolver los datos del calendario
@app.route('/api/calendario', methods=['GET'])
def api_calendario():
    fecha = request.args.get('fecha')
    if not fecha:
        return jsonify({"error": "Debe seleccionar una fecha válida."}), 400

    try:
        partidos = obtener_partidos_por_fecha(fecha)
        if partidos:
            return jsonify(partidos)
        else:
            return jsonify({"message": "No hay partidos disponibles para la fecha seleccionada."})
    except Exception as e:
        print(f"Error al obtener partidos para la fecha {fecha}: {str(e)}")
        return jsonify({"error": "Ocurrió un error al buscar los partidos. Intente nuevamente."}), 500

# API para devolver los datos del scoreboard
@app.route('/api/scoreboard', methods=['GET'])
def api_scoreboard():
    fecha_actual = datetime.today().strftime('%Y%m%d')
    partidos = obtener_partidos_por_fecha(fecha_actual)
    for partido in partidos:
        partido.update({
            "era_local": np.random.uniform(3.0, 5.0),  # Simulado
            "ops_local": np.random.uniform(0.700, 0.900),  # Simulado
            "whip_local": np.random.uniform(1.0, 1.5),  # Simulado
            "era_visitante": np.random.uniform(3.0, 5.0),  # Simulado
            "ops_visitante": np.random.uniform(0.700, 0.900),  # Simulado
            "whip_visitante": np.random.uniform(1.0, 1.5)  # Simulado
        })
    return jsonify(partidos)

# API para devolver los partidos en formato JSON (para usar con AJAX)
@app.route('/api/partidos', methods=['GET'])
def api_partidos():
    fecha_actual = datetime.today().strftime('%Y%m%d')
    partidos_hoy = obtener_partidos_por_fecha(fecha_actual)
    return jsonify(partidos_hoy)

@app.route('/pago')
def pago():
    return redirect('https://www.paypal.com/paypalme/tu_usuario/10')

if __name__ == '__main__':
    app.run(debug=True)
