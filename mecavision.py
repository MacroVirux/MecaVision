import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request
from pyzbar.pyzbar import decode as decode_barcode
from pylibdmtx.pylibdmtx import decode as decode_datamatrix
import easyocr
from PIL import Image
import threading
import logging
from queue import Queue, Empty
import time
import Levenshtein
from flask_cors import CORS
import re

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)

# Inicializar el lector OCR (EasyOCR)
try:
    ocr_reader = easyocr.Reader(['es', 'en'], gpu=True)
    logging.info("EasyOCR inicializado con soporte para GPU.")
except Exception as e:
    ocr_reader = easyocr.Reader(['es', 'en'], gpu=False)
    logging.warning(f"EasyOCR inicializado sin soporte para GPU. Error: {e}")

# Variables globales
RESIZE_WIDTH = 320
QUEUE_SIZE = 50
DESIRED_FPS = 5
FRAME_INTERVAL = 1.0 / DESIRED_FPS

frame_queue = Queue(maxsize=QUEUE_SIZE)             # Cola para frames sin procesar
processed_queue = Queue(maxsize=QUEUE_SIZE)         # Cola para frames procesados
capturando = False
resultados_udi = None                          # Resultados de la imagen UDI cargada
resultados_camara = {}                         # Resultados del análisis de la cámara
comparacion_resultados = {}                    # Resultados de la comparación
lock = threading.Lock()                        # Para sincronización

# Variables para manejo de hilos
cap = None
last_ocr_time = 0
OCR_INTERVAL = 2  # Realizar OCR cada 2 segundos

def preprocesar_imagen(image):
    """Preprocesar la imagen para mejorar la detección de códigos."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Equalización del histograma para mejorar el contraste
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    return blurred

def verificar_decodabilidad(image):
    """Verificar la decodabilidad de códigos QR y DataMatrix en la imagen."""
    # Decodificar códigos QR usando pyzbar
    qr_codes = decode_barcode(image)
    qr_data = [code.data.decode('utf-8') for code in qr_codes]

    # Decodificar códigos DataMatrix usando pylibdmtx
    datamatrix_codes = decode_datamatrix(image, timeout=1000, max_count=10)
    dm_data = [code.data.decode('utf-8') for code in datamatrix_codes]

    decodability = "PASS" if len(qr_codes) > 0 or len(datamatrix_codes) > 0 else "FAIL"
    return decodability, qr_codes, datamatrix_codes, qr_data, dm_data

def mostrar_resultados_en_frame(frame, resultados, qr_codes, datamatrix_codes):
    """Superponer resultados y detecciones en el frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    color_texto = (255, 255, 255)  # Blanco para texto
    color_fondo = (0, 0, 0)        # Negro semitransparente para fondo del texto
    thickness = 2

    # Dibujar rectángulos alrededor de códigos QR detectados
    if qr_codes is not None:
        for obj in qr_codes:
            puntos = obj.polygon
            if len(puntos) == 4:
                pts = np.array([(p.x, p.y) for p in puntos], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=3)

    # Dibujar rectángulos alrededor de códigos DataMatrix detectados
    if datamatrix_codes is not None:
        for obj in datamatrix_codes:
            rect = obj.rect
            x, y, w, h = rect.left, rect.top, rect.width, rect.height
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Superponer los resultados del análisis de decodabilidad
    if resultados:
        texto_lista = [
            f"Decodabilidad: {resultados.get('Decodability', 'N/A')}",
        ]
        for i, line in enumerate(texto_lista):
            y_position = 30 + i * 30
            # Dibujar fondo semitransparente
            (text_width, text_height), _ = cv2.getTextSize(line, font, 0.6, thickness)
            cv2.rectangle(frame, (10, y_position - text_height - 10), (10 + text_width + 10, y_position + 5), color_fondo, -1)
            # Dibujar texto
            cv2.putText(frame, line, (15, y_position), font, 0.6, color_texto, thickness)

    return frame

def procesar_frames():
    """Procesar frames desde la cola de captura y ponerlos en la cola de procesados."""
    global resultados_camara, capturando, last_ocr_time

    while capturando or not frame_queue.empty():
        try:
            frame = frame_queue.get(timeout=1)  # Esperar hasta 1 segundo por un frame
        except Empty:
            continue

        start_time = time.time()  # Inicio del procesamiento

        # Preprocesar la imagen
        preprocesada = preprocesar_imagen(frame)

        # Controlar la frecuencia de OCR
        current_time = time.time()
        realizar_ocr = False

        if current_time - last_ocr_time >= OCR_INTERVAL:
            realizar_ocr = True
            last_ocr_time = current_time

        # Verificar decodabilidad usando la imagen preprocesada
        decodability, qr_codes, datamatrix_codes, qr_data, dm_data = verificar_decodabilidad(preprocesada)
        resultados = {
            'Decodability': decodability,
            'QR_Data': qr_data,
            'DM_Data': dm_data,
            'OCR': []
        }

        # Realizar OCR si es necesario
        if realizar_ocr:
            ocr_results = ocr_reader.readtext(frame, detail=0, paragraph=True)
            resultados['OCR'] = [texto.lower() for texto in ocr_results]

        logging.info(f"Procesando frame: Decodabilidad={resultados['Decodability']}")

        # Actualizar resultados_camara
        resultados_camara = resultados

        # Superponer los resultados en el frame
        frame = mostrar_resultados_en_frame(frame, resultados, qr_codes, datamatrix_codes)

        # Poner el frame procesado en la cola de procesados para la transmisión
        processed_queue.put(frame)

        end_time = time.time()  # Fin del procesamiento
        processing_time = end_time - start_time
        logging.info(f"Tiempo de procesamiento del frame: {processing_time:.4f} segundos")

def capturar_frames_func(video_url):
    """Capturar frames desde la cámara y ponerlos en la cola de captura para procesar."""
    global cap, capturando

    cap = cv2.VideoCapture(video_url)

    if not cap.isOpened():
        logging.error("No se pudo acceder a la cámara. Verifique la conexión y la URL.")
        capturando = False
        return

    logging.info(f"Conectado a la cámara en {video_url}.")

    last_frame_time = time.time()

    while capturando:
        current_time = time.time()
        elapsed_time = current_time - last_frame_time

        if elapsed_time >= FRAME_INTERVAL:
            ret, frame = cap.read()
            if not ret:
                logging.warning("No se pudo leer el frame de la cámara.")
                time.sleep(0.01)
                continue

            last_frame_time = current_time

            # Redimensionar el frame para mejorar el rendimiento
            frame = cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * RESIZE_WIDTH / frame.shape[1])))

            # Poner el frame en la cola de captura (bloqueante)
            frame_queue.put(frame)
        else:
            # Esperar el tiempo restante
            time.sleep(FRAME_INTERVAL - elapsed_time)

    cap.release()
    logging.info("Captura de frames detenida.")

def iniciar_threads(video_url):
    """Iniciar los hilos de captura y procesamiento."""
    global capturando

    capturando = True

    # Iniciar hilo de captura
    hilo_captura = threading.Thread(target=capturar_frames_func, args=(video_url,), daemon=True)
    hilo_captura.start()

    # Iniciar hilos de procesamiento
    num_hilos_procesamiento = 2  # Ajusta según los núcleos disponibles
    for _ in range(num_hilos_procesamiento):
        hilo_procesamiento = threading.Thread(target=procesar_frames, daemon=True)
        hilo_procesamiento.start()

@app.route('/')
def index():
    """Ruta para servir la página principal."""
    return render_template('frontend.html')

@app.route('/video_feed')
def video_feed():
    """Ruta para el flujo de video en tiempo real."""
    def generar_frames():
        while capturando or not processed_queue.empty():
            try:
                # Intentar obtener el frame procesado más reciente
                frame = processed_queue.get(timeout=1)

                # Codificar el frame a JPEG con calidad reducida
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])  # Calidad al 70%
                if not ret:
                    logging.warning("No se pudo codificar el frame.")
                    continue
                frame_bytes = buffer.tobytes()

                # Enviar el frame al frontend
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error al generar frames: {e}")
                continue
        # Cuando capturando es False, salir del generador
        logging.info("Deteniendo transmisión de video.")
        return

    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cargar_udi', methods=['POST'])
def cargar_udi():
    """Cargar y procesar el archivo UDI."""
    global resultados_udi, comparacion_resultados

    archivo = request.files.get('udi_file')
    if not archivo:
        return jsonify({"error": "No se cargó ningún archivo."}), 400

    try:
        # Leer la imagen directamente
        pil_image = Image.open(archivo).convert('RGB')
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logging.error(f"Error al procesar el archivo: {e}")
        return jsonify({"error": f"Error al procesar el archivo: {e}"}), 500

    # Reducir la resolución del UDI para mejorar el rendimiento
    image = cv2.resize(image, (RESIZE_WIDTH, int(image.shape[0] * RESIZE_WIDTH / image.shape[1])))

    # Realizar OCR en el UDI
    ocr_results = ocr_reader.readtext(image, detail=0, paragraph=True)
    ocr_text = [texto.lower() for texto in ocr_results]

    # Decodificar códigos en el UDI
    preprocesada = preprocesar_imagen(image)
    decodability, qr_codes, datamatrix_codes, qr_data, dm_data = verificar_decodabilidad(preprocesada)

    resultados = {
        'Decodability': decodability,
        'QR_Data': qr_data,
        'DM_Data': dm_data,
        'OCR': ocr_text
    }

    logging.info("UDI cargado y procesado exitosamente.")

    # Almacenar los resultados del UDI
    resultados_udi = resultados

    # Resetea la comparación si se vuelve a cargar un UDI
    with lock:
        comparacion_resultados = {}

    return jsonify({
        "message": "UDI cargado exitosamente.",
        "resultados": resultados
    }), 200

@app.route('/iniciar_escaneo', methods=['POST'])
def iniciar_escaneo():
    """Iniciar el escaneo desde la cámara."""
    global capturando

    if capturando:
        return jsonify({"error": "El escaneo ya está en curso."}), 400

    # Obtener la dirección IP de la cámara desde los datos del formulario
    ip = request.form.get('camera_ip')
    if not ip:
        return jsonify({"error": "No se proporcionó la dirección IP de la cámara."}), 400

    # Limpiar y validar la dirección IP
    ip = ip.strip()

    # Validar formato de dirección IP
    ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
    if not ip_pattern.match(ip):
        return jsonify({"error": "Dirección IP de la cámara inválida."}), 400

    # Construir la URL de la cámara
    puerto = request.form.get('camera_port', default=8080, type=int)  # Permite especificar el puerto, por defecto 8080
    video_url = f"http://{ip}:{puerto}/video"  # Reemplaza con la URL correcta

    logging.info(f"URL construida: {video_url}")

    # Probar la captura de video
    test_cap = cv2.VideoCapture(video_url)
    if not test_cap.isOpened():
        logging.error("No se pudo acceder a la cámara. Verifique la conexión y la URL.")
        return jsonify({"error": "No se pudo acceder a la cámara. Verifique la conexión y la URL."}), 500
    test_cap.release()

    # Iniciar los hilos de captura y procesamiento
    iniciar_threads(video_url)

    logging.info("Escaneo iniciado.")
    return jsonify({"message": "Escaneo iniciado."}), 200

@app.route('/detener_escaneo', methods=['POST'])
def detener_escaneo():
    """Detener el escaneo."""
    global capturando, resultados_camara, comparacion_resultados
    if not capturando:
        return jsonify({"error": "No hay escaneo en curso."}), 400

    capturando = False

    # Vaciar la cola de captura
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except:
            break

    # Vaciar la cola de procesados
    while not processed_queue.empty():
        try:
            processed_queue.get_nowait()
        except:
            break

    # Resetear resultados_camara y comparacion_resultados
    resultados_camara = {}
    with lock:
        comparacion_resultados = {}

    logging.info("Escaneo detenido.")
    return jsonify({"message": "Escaneo detenido."}), 200

@app.route('/obtener_resultados_udi', methods=['GET'])
def obtener_resultados_udi():
    """Obtener los resultados del UDI cargado."""
    global resultados_udi
    if resultados_udi:
        return jsonify({
            "resultados": resultados_udi
        }), 200
    else:
        return jsonify({"resultados": None}), 200

@app.route('/obtener_resultados_camara', methods=['GET'])
def obtener_resultados_camara():
    """Obtener los resultados del análisis de la cámara en tiempo real."""
    global resultados_camara
    if resultados_camara:
        return jsonify({
            "resultados": resultados_camara
        }), 200
    else:
        return jsonify({"resultados": None}), 200

@app.route('/comparar_resultados', methods=['GET'])
def obtener_comparacion():
    """Obtener las comparaciones entre UDI y cámara."""
    global resultados_udi, resultados_camara, comparacion_resultados

    with lock:
        comparacion_resultados = {}

        if resultados_udi and resultados_camara:
            # Comparar Decodabilidad
            comparacion_resultados['Decodabilidad'] = resultados_udi.get('Decodability') == resultados_camara.get('Decodability')

            # Comparar QR_Data
            udi_qr = ' '.join(resultados_udi.get('QR_Data', []))
            cam_qr = ' '.join(resultados_camara.get('QR_Data', []))
            comparacion_resultados['QR_Data'] = udi_qr == cam_qr

            # Comparar DM_Data
            udi_dm = ' '.join(resultados_udi.get('DM_Data', []))
            cam_dm = ' '.join(resultados_camara.get('DM_Data', []))
            comparacion_resultados['DM_Data'] = udi_dm == cam_dm

            # Comparar OCR usando distancia de Levenshtein
            udi_ocr = ' '.join(resultados_udi.get('OCR', []))
            camara_ocr = ' '.join(resultados_camara.get('OCR', []))
            distancia = Levenshtein.distance(udi_ocr.lower(), camara_ocr.lower())
            max_length = max(len(udi_ocr), len(camara_ocr))
            similarity = (max_length - distancia) / max_length if max_length > 0 else 1
            comparacion_resultados['OCR'] = similarity >= 0.8  # Considerar similar si hay un 80% de coincidencia

            # Agregar más comparaciones si es necesario

            return jsonify({"comparacion": comparacion_resultados}), 200
        else:
            return jsonify({"error": "No hay resultados para comparar. Asegúrese de haber cargado el UDI y de tener resultados de la cámara."}), 400

if __name__ == "__main__":
    # Asegurar que las colas estén vacías al iniciar
    while not frame_queue.empty():
        frame_queue.get_nowait()
    while not processed_queue.empty():
        processed_queue.get_nowait()

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
