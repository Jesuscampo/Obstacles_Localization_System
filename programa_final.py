# Importe de Librerías
import cv2
import RPi.GPIO as GPIO
import time

from gtts import gTTS           # Convertir texto a Voz / Modulos
from playsound import playsound # Reproducción de audio

import threading    # Paralelismo
import queue
import random




# DETECCIÓN DE OBJETOS

# Se almacena el nombre de todas las clases que el modelo es capaz de detectar
classNames = []
classFile = "/home/chuchicampo/Documentos/TrabajoFinal/spanish.names"   # Se obtienen del fichero 'spanish.names'

with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")
    
# Se almacenan las clases de los objetos que quieren ser detectados, es este caso, los potenciales obstáculos
obs = []
obsFile = "/home/chuchicampo/Documentos/TrabajoFinal/obstacles.names"   # Se obtienen del fichero 'obstacles.names'

with open(obsFile,"rt") as f:
    obs = f.read().rstrip("\n").split("\n")


# Rutas de los ficheros de configuración y pesos
configPath = "/home/chuchicampo/Documentos/TrabajoFinal/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/chuchicampo/Documentos/TrabajoFinal/frozen_inference_graph.pb"

# Creación de la red neuronal en base a la configuración y pesos definidos
net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,240)       # Tamaño de entrada de la imagen en la red neuronal
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))    
net.setInputSwapRB(True)

# Umbral de coincidencia para detectar un objeto
thres = 0.5   

# Coordenadas de la caja que delimita el marco de proyección del sensor de ultrasonidos sobre la vista de la cámara
limit = [180, 135, 280, 210]





# ESTABLECIMIENTO DE PARAMETROS DEL SENSOR DE ULTRASONIDOS
# Se establece el modo del GPIO
GPIO.setmode(GPIO.BCM)

# Número de los Pines Trigger y Echo
GPIO_TRIGGER = 8
GPIO_ECHO = 10

# Se establece el pin Trigger como salida y Echo como entrada
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)




def getObjects(img, thres, nms, objects=[]):
    
    '''
    Esta función recibe como entrada un fotograma capturado por video a través de la cámara y
    aplica sobre la misma el modelo de detección de objetos, donde se identifican todos los objetos
    existentes en la imagen que superen el umbral de confianza. A su vez, se dibuja una caja que delimita
    el espacio que ocupa dicho objeto junto al nombre del objeto y un punto que marca el centro de dicha caja.
    
    Args:
        img: Imagen que va a ser procesada por el modelo de detección
        thres: Indica el valor mínimo de confianza de un objeto para que sea detectado
        nms: Indica cuanta supresión de no máximos se aplicará a las detecciones
        objects: Lista de clases de objetos que se pueden detectar
    
    Returns:
        img: Imagen ya procesada con cada objeto detectado con sus respectivas cajas delimitadoras y su nombre dibujados
        objectName: Lista con los nombres de los objetos detectados 
        inside: Lista con valores 1 y 0, que indican si el objeto esta dentro de la caja limite 
                de detección (1) o no (0), donde el orden de los valores corresponde con el orden 
                de los objetos en la lista 'objectName'
        size: Lista con las áreas que ocupan en la imagen cada objeto, siguiendo el mismo orden que
              las listas 'objectName' y 'inside'
    '''
    
    
    idsClases, tasasConf, cajas = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectName =[]
    inside = []
    size = []
    if len(idsClases) != 0:
        for idClase, conf,caja in zip(idsClases.flatten(),tasasConf.flatten(),cajas):
            className = classNames[idClase - 1]
            if className in objects:
                objectName.append(className)

                # Coordenadas del centro del objeto
                x = (2*caja[0] + caja[2]) / 2
                y = (2*caja[1] + caja[3]) / 2
                    
                # Caja delimitadora del objeto
                cv2.rectangle(img,caja,color=(0,255,0),thickness=2)
                # Punto central del objeto
                cv2.circle(img,(int(x), int(y)),1,color=(255,0,0),thickness=2)
                # Caja límite de detección
                cv2.rectangle(img,limit,color=(0,0,255),thickness=2)
                    
                # Se escribe el nombre del objeto y el porcentaje de coincidencia
                cv2.putText(img,classNames[idClase-1].upper(),(caja[0]+10,caja[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(img,str(round(conf*100,2)),(caja[0]+200,caja[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    
                # Se comprueba si el centro del objeto esta dentro de la caja limite de deteccion
                if x >= limit[0] and x <= (limit[0] + limit[2]) and y >= limit[1] and y<= (limit[1] + limit[3]):
                    inside.append(1)    # El 1 significa verdadero
                    size_obj = caja[2]*caja[3]    # En caso afirmativo, se calcula el área de ocupación aproximada
                    size.append(size_obj)
                else:
                    inside.append(0)    # El 0 significa falso
                    size.append(0)

    return img,objectName,inside,size



def measurement():
    
    '''
    Esta función realiza el cálculo de la distancia hasta el obstáculo más próximo en
    base a la información recibida del sensor HC-SR04 mediante los pines GPIO. 
    
    Args: NONE
    
    Returns: 
        distance: Distancia calculada en centímetros
    
    
    '''
    
	# Se establece el pin TRIGGER en estado ALTO
    GPIO.output(GPIO_TRIGGER, True)

	# Se espera 0.01ms y se establece el pin TRIGGER en estado BAJO, para enviar el pulso ultrasónico
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)

    # Se inicializa el tiempo de salida y llegada del pulso ultrasónico
    StartTime = time.time()
    StopTime = time.time()

	# Cuando el pin ECHO pase a estado ALTO, se guarda el tiempo de salida,
    # ya que significará que el pulso ultrasónico ha salido
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()

	# Cuando el pin ECHO pase a estado BAJO, se guarda el tiempo de llegada,
    # ya que significará que el pulso ultrasónico ha llegado
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()

	# Se calcula el tiempo transcurrido entre salida y llegada del pulso ultrasónico
    TotalTime = StopTime - StartTime
    
    
	# Se calcula la distancia, multiplicando el tiempo por la velocidad del sonido (34300 cm/s)
    # y dividido entre dos, porque el tiempo es de ida y vuelta.
    distance = (TotalTime * 34300) / 2

    return distance


def capturador():
    
    '''
    Es la función principal del 'Hilo Capturador' y captura vídeo mediante la Camera PI y añade los 
    fotogramas sobre una cola pública sobre la cual tendrá acceso el 'Hilo Procesador'
    
    Args: NONE
    
    Returns: NONE 
    
    '''
    cap = cv2.VideoCapture(0)   # Se comienza a capturar video
    cap.set(3,640)
    cap.set(4,480)

    while True:
        # Se lee un fotograma y se añade a la cola 'frames'
        success, img = cap.read()
        frames.put(img)     
        ev1.set()   # Se notifica al 'Hilo Procesador'


    
def procesador():
    
    '''
     Es la función principal del 'Hilo Procesador' y procesa cada fotograma que lee de la cola 'frames'
     aplicando la función 'getObjects', y a su vez calcula la distancia al objeto más próximo mediante el uso
     de la función 'measurement'. Determina que objeto es el obstáculo más próximo, y dicha información junto
     a la distancia en metros, se añade sobre una cola sobre la cual tendrá acceso el 'Hilo Reproductor'.
    
    Args: NONE
    
    Returns: NONE
    
    
    '''
    
    while True:
        ev1.wait()
        size_max = 0
        obj = None
        
        if not frames.empty():
            # Se lee el último fotograma añadido a la cola 'frames' y se aplica la detección de objetos
            img = frames.get()
            result, names, inside, size = getObjects(img,thres,0.2, objects=obs)
            
            cv2.imshow("Output", img)
            
            # De entre todos los objetos detectados, se determina el obstáculo más próximo dentro del rango del sensor
            for i, s, o in zip(inside, size, names):
                if (i == 1) and s > size_max:
                    size_max = s
                    obj = o
            
            # Se calcula la distancia y se redondea en metros
            dist = measurement()
            meters = round(int(dist) / 100, 1)
            
            # Se añaden la distancia y el nombre del obstáculo a sus respectivas colas
            distances.put(meters)
            obstacles.put(obj)
            
            # Se notifica al 'Hilo Reproductor' y se espera a que finalice
            ev2.set()
            ev3.wait()
            
            # Se limpia la cola de fotogramas
            frames.queue.clear()
            
            ev3.clear()
            cv2.waitKey(1)
        
        ev1.clear()


def reproductor():
    
    '''
    Es la función principal del 'Hilo Reproductor' y lee de una cola la información relativa sobre la 
    distancia al obstáculo más próximo. Crea un mensaje de voz con dicha información y lo reproduce 
    a través de un altavoz.
    
    Args: NONE
    
    Returns: NONE
    '''
    
    while True:
        ev2.wait()
        if not distances.empty():
            # Se lee la última distancia calculada y el último obstáculo detectado
            distancia = distances.get()
            obs = obstacles.get()
            
            # En caso de no haberse detectado ningún objeto, se considerará un 'obstáculo' cualquiera
            if not obs:
                nombreObs = "un obstáculo"
            else:
                nombreObs = obs

            # En caso de que la distancia sea inferior a 5 metros, se notifica al usuario del obstáculo
            if distancia < 5:
                audio = gTTS(f"{distancia} metros a {nombreObs}", lang='es')
                audio.save('/home/chuchicampo/Documentos/TrabajoFinal/audio.mp3')
                playsound('/home/chuchicampo/Documentos/TrabajoFinal/audio.mp3')
            
            nombreObs = None
  
            ev3.set()

        ev2.clear()


if __name__ == "__main__":
    
    # Inicialización de las colas
    frames = queue.Queue()      # Almacena fotogramas capturados por la cámara
    distances = queue.Queue()   # Almacena distancias calculadas por el sensor
    obstacles = queue.Queue()   # Almacena nombres de los obstáculos detectados
        
    # Inicialización de eventos
    ev1 = threading.Event()
    ev2 = threading.Event()
    ev3 = threading.Event()

    # Inicialización de los hilos
    hilo_capturador = threading.Thread(target=capturador)
    hilo_procesador = threading.Thread(target=procesador)
    hilo_reproductor = threading.Thread(target=reproductor)
    
    # Comienza la ejecución de los hilos
    hilo_capturador.start()
    hilo_procesador.start()
    hilo_reproductor.start()
    
    # Sincronización de la ejecución de los hilos
    hilo_capturador.join()
    hilo_procesador.join()
    hilo_reproductor.join()
      

        
