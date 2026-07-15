import time
import cv2
import numpy as np
import threading

print("Iniciando sistema Multi-threading Asíncrono (Fijado a 640x640)...")
net = cv2.dnn.readNetFromONNX("best.onnx")

CLASES = ["Bata", "Casco", "Gafas", "Persona"]
EPPS_OBLIGATORIOS = ["Casco", "Gafas", "Bata"]

frame_actual = None
personas_detectadas = []
corriendo = True
lock = threading.Lock()

def calcular_interseccion(box_epp, box_persona):
    xA, yA = max(box_epp[0], box_persona[0]), max(box_epp[1], box_persona[1])
    xB, yB = min(box_epp[2], box_persona[2]), min(box_epp[3], box_persona[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0: return 0.0
    area_epp = (box_epp[2] - box_epp[0]) * (box_epp[3] - box_epp[1])
    return inter_area / area_epp

# ==========================================
# HILO SECUNDARIO: Motor de Inferencia YOLO
# ==========================================
def hilo_procesamiento_yolo():
    global frame_actual, personas_detectadas, corriendo
    print("Hilo de IA (YOLO) iniciado con éxito.")
    
    while corriendo:
        with lock:
            if frame_actual is None:
                continue
            frame_procesar = frame_actual.copy()

        alto, ancho, _ = frame_procesar.shape
        max_dim = max(alto, ancho)
        img_cuadrada = np.zeros((max_dim, max_dim, 3), np.uint8)
        img_cuadrada[0:alto, 0:ancho] = frame_procesar

        blob = cv2.dnn.blobFromImage(img_cuadrada, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        
        try:
            salida = net.forward()[0].T 
        except cv2.error as e:
            print(f"Error interno en capa DNN: {e}")
            time.sleep(1)
            continue
        
        boxes, scores, class_ids = [], [], []

        for row in salida:
            clases_scores = row[4:]
            class_id = np.argmax(clases_scores)
            max_score = clases_scores[class_id]

            if max_score > 0.5:
                x_c, y_c, w, h = row[0], row[1], row[2], row[3]

                x_c = int(x_c * max_dim / 640)
                y_c = int(y_c * max_dim / 640)
                w = int(w * max_dim / 640)
                h = int(h * max_dim / 640)
                
                x1 = int(x_c - (w / 2))
                y1 = int(y_c - (h / 2))
                
                boxes.append([x1, y1, w, h])
                scores.append(float(max_score))
                class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4)
        
        personas_temp = []
        epps_temp = []

        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                x1, y1, w, h = box
                coords = [x1, y1, x1 + w, y1 + h]
                label = CLASES[class_ids[i]]
                
                if label == "Persona":
                    personas_temp.append({"box": coords, "epps": []})
                elif label in EPPS_OBLIGATORIOS:
                    epps_temp.append({"label": label, "box": coords})

        for epp in epps_temp:
            mejor_match, max_overlap = None, 0.3 
            for i, persona in enumerate(personas_temp):
                overlap = calcular_interseccion(epp["box"], persona["box"])
                if overlap > max_overlap:
                    max_overlap, mejor_match = overlap, i
            if mejor_match is not None:
                personas_temp[mejor_match]["epps"].append(epp["label"])

        with lock:
            personas_detectadas = personas_temp

        time.sleep(0.01)

# ==========================================
# HILO PRINCIPAL: Captura de Video e Interfaz
# ==========================================
print("Encendiendo cámara IMX708...")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(2)

hilo_yolo = threading.Thread(target=hilo_procesamiento_yolo)
hilo_yolo.start()

print("Interfaz gráfica asíncrona iniciada. Presiona 'q' para salir.")
tiempo_anterior = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.01)
        continue

    with lock:
        frame_actual = frame.copy()
        personas_dibujar = personas_detectadas.copy()

    tiempo_actual = time.time()
    fps_video = int(1 / (tiempo_actual - tiempo_anterior))
    tiempo_anterior = tiempo_actual

    for idx, persona in enumerate(personas_dibujar):
        p_box = persona["box"]
        faltantes = [epp for epp in EPPS_OBLIGATORIOS if epp not in persona["epps"]]
        
        if len(faltantes) == 0:
            color, mensaje = (0, 255, 0), "EPP Completo"
        else:
            color, mensaje = (0, 0, 255), f"FALTA: {', '.join(faltantes)}"

        cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), color, 2)

        if p_box[1] < 25:
            y_texto = p_box[1] + 20
        else:
            y_texto = p_box[1] - 10
            
        cv2.putText(frame, f"Persona {idx+1}: {mensaje}", (p_box[0], y_texto), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    cv2.putText(frame, f"Video GUI FPS: {fps_video}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Control de EPP - Multi-threading", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        corriendo = False 
        break

hilo_yolo.join()
cap.release()
cv2.destroyAllWindows()
print("Programa finalizado correctamente.")
