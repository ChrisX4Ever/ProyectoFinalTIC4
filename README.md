# Detección de EPP en Tiempo Real con YOLO ONNX y Raspberry Pi 4B
Trabajo Final de TIC 4  
Nuestro tema es Proyecto 2: Monitoreo de ocupación y conducta insegura en laboratorio o taller  
Integrantes:  
*   **Pedro Nuñez Moraga**
*   **Christian Silva Pinilla**    
[Video Explicativo](https://www.youtube.com/watch?v=cPu4cDjBeb4)  

Este repositorio contiene el sistema de inspección y validación automatizada de **Equipos de Protección Personal (EPP)** optimizado para ejecutarse en el borde (*Edge AI*) utilizando una **Raspberry Pi 4B** y el módulo de cámara **ArduCam IMX708**.

El modelo está diseñado y entrenado para detectar específicamente los siguientes elementos críticos de seguridad:
*   **Casco Blanco**
*   **Antiparras**
*   **Bata/Delantal**

Además, el software incluye la lógica algorítmica encargada de comprobar si los elementos de protección se encuentran posicionados correctamente sobre la **Persona** detectada, notificando en tiempo real si el equipo está completo o indicando con precisión cuáles componentes faltan.

---

## 🛠️ Requisitos de Hardware

*   **Placa:** Raspberry Pi 4B (Recomendado versión de 4GB o 8GB de RAM).
*   **Cámara:** ArduCam IMX708 (Cámara Raspberry Pi Módulo 3, con soporte de Enfoque Automático).
*   **Sistema Operativo:** Raspberry Pi OS (64-bit) Bookworm o posterior, con arquitectura de cámara moderna basada en `libcamera`.

---

## 📦 Instalación y Configuración

Sigue estos pasos en la terminal de tu Raspberry Pi para clonar el repositorio, configurar el entorno virtual de Python e instalar las dependencias requeridas.

### 1. Clonar el Repositorio
Para descargar el script principal `detector_epp.py`, el modelo optimizado `best.onnx` y la documentación asociada, ejecuta:
```bash
git clone https://github.com/ChrisX4Ever/ProyectoFinalTIC4.git
cd ProyectoFinalTIC4
```

### 2. Actualizar el Sistema e Instalar Librerías Base
Es necesario asegurar la presencia de las herramientas de compilación y las librerías compartidas que OpenCV requiere para el procesamiento de imágenes en entornos Linux:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv libgl1-mesa-glx libglib2.0-0
```

### 3. Crear y Activar un Entorno Virtual de Python
Para evitar conflictos con los paquetes administrados por el gestor de software del sistema (`apt`):
```bash
python3 -m venv env
source env/bin/activate
```

### 4. Instalar Dependencias del Proyecto
Instala los paquetes de Python necesarios para decodificar video, interactuar con matrices numéricas y correr la inferencia del modelo ONNX de forma eficiente:
```bash
pip install --upgrade pip
pip install opencv-python numpy onnxruntime
```

---

## 🚀 Ejecución del Programa

Para inicializar la cámara ArduCam IMX708 utilizando el entorno de compatibilidad nativa de la Raspberry Pi y ejecutar el script de análisis en tiempo real, corre el siguiente comando en tu terminal:

```bash
libcamerafy python detector_epp.py --model best.onnx
```

### Parámetros Adicionales Configurables:
El script permite modificar umbrales directamente desde la línea de comandos para facilitar las pruebas de campo:
```bash
libcamerafy python detector_epp.py --model best.onnx --conf 0.50
```
*   `--model`: Especifica la ruta del modelo (`best.onnx` por defecto).
*   `--conf`: Umbral mínimo de confianza para filtrar detecciones falsas (ej: `0.50` para un 50%).

> 💡 **Nota sobre `libcamerafy`:** El uso de esta herramienta como prefijo garantiza la correcta inyección del pipeline moderno de `libcamera` en el script de Python, permitiendo que OpenCV capture los fotogramas del sensor IMX708 preservando el autoenfoque por hardware y las optimizaciones de color/exposición automáticas.

---

## 🧠 Lógica de Inferencia y Validación

El script `detector_epp.py` no se limita a lanzar alertas de objetos flotantes en la imagen, sino que ejecuta un algoritmo de **agrupación espacial por intersección**:

1.  **Detección de Sujeto:** Localiza la caja delimitadora (*Bounding Box*) que encierra a una `Persona`.
2.  **Análisis de Inclusión:** Evalúa si las cajas correspondientes a `Casco_Blanco`, `Antiparras`, `Chaqueta_Reflectante` y `Guantes` están contenidas geométricamente (o intersectan significativamente) dentro de las regiones anatómicas lógicas del recuadro de la persona.
3.  **Evaluación de Seguridad:**
    *   **EPP Completo ✅:** Se detectan los 4 elementos de seguridad sobre el individuo. El sistema dibuja un marco verde de aprobación.
    *   **EPP Incompleto ❌:** Si falta al menos uno de los elementos, el sistema genera una alerta en la interfaz visual y escribe en la terminal la lista exacta de lo que falta (Ejemplo: `FALTAN: Antiparras, Guantes de Seguridad`).

---

## 📊 Rendimiento y Limitaciones Técnicas

*   **Motor de Inferencia:** Ejecutado puramente en la CPU (ARM Cortex-A72 de la RPi 4B) a través de `onnxruntime`.
*   **Resolución de Entrada:** Redimensionado automático a **640x640 píxeles** para preservar la capacidad de reconocer objetos pequeños (como las antiparras).
*   **Tasa de Cuadros (FPS):** Debido a la ausencia de un acelerador neural de hardware dedicado, el rendimiento oscilará entre los **2 y 5 FPS**.
*   **Caso de Uso Recomendado:** **Puntos de Control Estáticos**. Está diseñado idealmente para accesos peatonales obligatorios, molinetes/torniquetes o accesos a talleres, donde el personal se detiene frente a la cámara por 1-2 segundos para verificar su equipamiento antes de ingresar a la zona de riesgo.

---

## 📈 Robustez ante Iluminación
El modelo `best.onnx` incluido en este repositorio fue entrenado aplicando técnicas avanzadas de aumentación de datos en **Roboflow**, lo que le otorga resistencia frente a:
*   Entornos con sobreexposición lumínica (luz solar directa de mediodía).
*   Bajos niveles de iluminación, sombras densas y penumbras artificiales.
*   Ruido digital o "grano" en la imagen provocado por el sensor de la cámara en capturas nocturnas.
