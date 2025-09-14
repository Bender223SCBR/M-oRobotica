import mediapipe
import scipy.spatial
import threading
import multiprocessing
import numpy as np
import serial
import serial.tools.list_ports
import time
import cv2

# Lock for thread synchronization
lock = threading.Lock()

# Number of arrays and array size
num_arrays = 6
array_size = 4

# Font for text
font = cv2.FONT_HERSHEY_SIMPLEX

# Video capture
capture = cv2.VideoCapture(0)
frameWidth = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Modules and libraries for image processing and hand recognition
drawingModule = mediapipe.solutions.drawing_utils
handsModule = mediapipe.solutions.hands
distanceModule = scipy.spatial.distance

# Colors
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
black = (0, 0, 0)
white = (255, 255, 255)
lightBlue = (255, 255, 0)
colors = [lightBlue, blue, green, red, yellow, white, black]

# Coordinates of the center and radius of the circle
center_x, center_y = int(frameWidth / 2), int(frameHeight / 2)
circle_center = (center_x, center_y)
circle_radius = int(frameHeight / 4)

# Shared variable to check if all hand is inside the circle
all_inside = multiprocessing.Value('b', False)

# Shared arrays for hand coordinates, distances, and proportional distances
finger_coordinates = [multiprocessing.Array('f', array_size) for _ in range(num_arrays)]
distances = [multiprocessing.Value('i', 0) for _ in range(num_arrays)]
proportional_distances = [multiprocessing.Value('i', 0) for _ in range(num_arrays - 1)]


def handInCircle(RESULTS, ALL_INSIDE, FRAME):
    all_inside_check = False

    for LANDMARKS in RESULTS.multi_hand_landmarks:

        # all_inside, elapsed_time = check_hand()
        for LANDMARK in LANDMARKS.landmark:
            x, y = int(LANDMARK.x * FRAME.shape[1]), int(LANDMARK.y * FRAME.shape[0])

            distance_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5

            if distance_center > circle_radius:
                all_inside_check = False
                break

            else:
                all_inside_check = True

    ALL_INSIDE.value = all_inside_check


def drawFingers(FRAME, FINGER_COORDINATES, CIRCLE_CENTER, CIRCLE_RADIUS, ALL_INSIDE, ON, PROPORTIONAL_DISTANCES,
                START_TIME):
    if ON:
        ordinates = [[] for _ in range(len(FINGER_COORDINATES))]

        for i in range(len(FINGER_COORDINATES)):
            for j in range(0, len(FINGER_COORDINATES[i]), 2):
                if j + 1 < len(FINGER_COORDINATES[i]):
                    tupla = (int(FINGER_COORDINATES[i][j]), int(FINGER_COORDINATES[i][j + 1]))
                    ordinates[i].append(tupla)

        # Draw
        for i in range(len(ordinates)):
            if len(ordinates[i]) == 2:
                cv2.line(FRAME, ordinates[i][0], ordinates[i][1], colors[i], 1)

                middle_x = (ordinates[i][0][0] + ordinates[i][1][0]) // 2
                middle_y = (ordinates[i][0][1] + ordinates[i][1][1]) // 2
                try:
                    cv2.putText(FRAME, str(round(PROPORTIONAL_DISTANCES[i])) + "%",
                                (middle_x, middle_y), font, 0.5, colors[i], 1, cv2.LINE_4)
                except IndexError:
                    pass
    else:
        cv2.circle(FRAME, circle_center, CIRCLE_RADIUS, (0, 0, 255), 2)

        if ALL_INSIDE.value == 1:
            try:
                ang = (time.time() - START_TIME)
                cv2.ellipse(FRAME, CIRCLE_CENTER, (CIRCLE_RADIUS, CIRCLE_RADIUS), -90, 0, ang * 120,
                            (0, 175, 0), 3)
                cv2.ellipse(FRAME, CIRCLE_CENTER, (CIRCLE_RADIUS, CIRCLE_RADIUS), -90, 0, ang * 120,
                            (0, 255, 0), 2)
            except TypeError:
                pass


def scale(inp, inpmax, inpmin, outmax, outmin):
    out = ((inp - inpmin) * (outmax - outmin) / (inpmax - inpmin)) + outmin
    # out = max(out, 0)
    return out


def pixelCoordinates(framewidth, frameheight, keypoint: mediapipe.solutions.hands.HandLandmark, defPxCResults):
    normalizedLandmark = defPxCResults.multi_hand_landmarks[0].landmark[keypoint]
    pixelcoordinates = drawingModule._normalized_to_pixel_coordinates(
        normalizedLandmark.x,
        normalizedLandmark.y,
        framewidth,
        frameheight)
    return pixelcoordinates


def coordinates(FINGER_COORDINATES, defCResults):
    global handsModule

    fingers = [[] for _ in range(len(FINGER_COORDINATES))]
    index_finger_coordinates, \
        middle_finger_coordinates, \
        ring_finger_coordinates, \
        pinky_finger_coordinates, \
        thumb_coordinates, \
        prop_coordinates = fingers

    for LANDMARK in handsModule.HandLandmark:

        if LANDMARK == handsModule.HandLandmark.INDEX_FINGER_TIP:
            index_finger_coordinates.append(pixelCoordinates(frameWidth, frameHeight, LANDMARK, defCResults))
            index_finger_coordinates.append(pixelCoordinates(frameWidth, frameHeight,
                                                             handsModule.HandLandmark.INDEX_FINGER_MCP, defCResults))
        elif LANDMARK == handsModule.HandLandmark.MIDDLE_FINGER_TIP:
            middle_finger_coordinates.append(pixelCoordinates(frameWidth, frameHeight, LANDMARK, defCResults))
            middle_finger_coordinates.append(pixelCoordinates(frameWidth, frameHeight,
                                                              handsModule.HandLandmark.MIDDLE_FINGER_MCP, defCResults))
        elif LANDMARK == handsModule.HandLandmark.RING_FINGER_TIP:
            ring_finger_coordinates.append(pixelCoordinates(frameWidth, frameHeight, LANDMARK, defCResults))
            ring_finger_coordinates.append(pixelCoordinates(frameWidth, frameHeight,
                                                            handsModule.HandLandmark.RING_FINGER_MCP, defCResults))
        elif LANDMARK == handsModule.HandLandmark.PINKY_TIP:
            pinky_finger_coordinates.append(pixelCoordinates(frameWidth, frameHeight, LANDMARK, defCResults))
            pinky_finger_coordinates.append(pixelCoordinates(frameWidth, frameHeight,
                                                             handsModule.HandLandmark.PINKY_MCP, defCResults))
        elif LANDMARK == handsModule.HandLandmark.THUMB_TIP:
            thumb_coordinates.append(pixelCoordinates(frameWidth, frameHeight, LANDMARK, defCResults))
            thumb_coordinates.append(pixelCoordinates(frameWidth, frameHeight,
                                                      handsModule.HandLandmark.PINKY_MCP, defCResults))
        elif LANDMARK == handsModule.HandLandmark.WRIST:
            prop_coordinates.append(pixelCoordinates(frameWidth, frameHeight, LANDMARK, defCResults))
            prop_coordinates.append(pixelCoordinates(frameWidth, frameHeight,
                                                     handsModule.HandLandmark.MIDDLE_FINGER_MCP, defCResults))

    for a in range(len(fingers)):
        for b in range(len(FINGER_COORDINATES[a])):
            try:
                i_str = str(bin(b)[2:]).zfill(2)
                lsb = int(i_str[0])
                msb = int(i_str[1])
                FINGER_COORDINATES[a][b] = fingers[a][lsb][msb]
            except TypeError:
                pass


def calculateDistance(FINGER_COORDINATES, DISTANCES, PROPORTIONAL_DISTANCES):
    for X in range(len(DISTANCES)):
        tuplas = []
        for Y in range(0, len(FINGER_COORDINATES[X]), 2):
            if Y + 1 < len(FINGER_COORDINATES[X]):
                defTupla = (FINGER_COORDINATES[X][Y], FINGER_COORDINATES[X][Y + 1])
                tuplas.append(list(defTupla))
        DISTANCES[X] = int(distanceModule.euclidean(tuplas[0], tuplas[1]))

    for i in range(len(PROPORTIONAL_DISTANCES)):
        try:
            PROPORTIONAL_DISTANCES[i] = DISTANCES[i] / DISTANCES[-1]
        except ZeroDivisionError:
            pass

        if i != 4:
            PROPORTIONAL_DISTANCES[i] = min(max(round(PROPORTIONAL_DISTANCES[i] * 100), 0), 100)
        else:
            PROPORTIONAL_DISTANCES[i] = min(max(round(scale(PROPORTIONAL_DISTANCES[i] * 100, 100, 20, 100, 0)), 0),
                                            100)


def calculus(RESULTS, FINGER_COORDINATES, DISTANCES, PROPORTIONAL_DISTANCES):
    with lock:
        p1 = threading.Thread(target=coordinates, args=(FINGER_COORDINATES, RESULTS))
        p2 = threading.Thread(target=calculateDistance,
                              args=(FINGER_COORDINATES, DISTANCES, PROPORTIONAL_DISTANCES))

        p1.start()
        p1.join()
        p2.start()
        p2.join()


def listar_portas_disponiveis():
    # Lista todas as portas de comunicação disponíveis
    portas = list(serial.tools.list_ports.comports())
    return portas


def selecionar_porta():
    # Lista todas as portas disponíveis
    portas = listar_portas_disponiveis()

    if not portas:
        print("Nenhuma porta de comunicação disponível.")
        return None

    print("Portas de comunicação disponíveis:")
    for i, porta in enumerate(portas, start=1):
        print(f"{i}. {porta.device}")

    while True:
        try:
            escolha = int(input("Escolha o número da porta que deseja usar: "))
            if 1 <= escolha <= len(portas):
                return portas[escolha - 1].device
            else:
                print("Escolha um número válido.")
        except ValueError:
            print("Entrada inválida. Digite um número válido.")


def arduino_thread(tupla_numeros):

    serial_port = selecionar_porta()
    time.sleep(1)

    while True:

        if not serial_port:
            break
        else:
            ser = serial.Serial(serial_port, 9600, timeout=1)
            time.sleep(1)

        try:

            while True:
                if tupla_numeros[0] >= 1:
                    tupla_str = ','.join(map(str, tupla_numeros))
                    ser.write(tupla_str.encode())
                    print(f"Enviado: {tupla_str}")

                    resposta = ser.readline().decode().strip()
                    print(f"Resposta do Arduino: {resposta}")

                    while resposta != "Dados recebidos com sucesso":
                        resposta = ser.readline().decode().strip()
                        print(f"Resposta do Arduino: {resposta}")

        except KeyboardInterrupt:
            pass

        except ValueError:
            pass

        except TypeError:
            pass

        except Exception as e:
            print(f"Err: {str(e)}")

        finally:
            aguardar = 1.5
            print(f"Esperando {aguardar} segundos para reiniciar...")
            time.sleep(aguardar)
            ser.close()


def cam(PROPORTIONAL_DISTANCES):
    # Control variable for the "on" or "off" state
    on = False
    start_time = time.time() - time.time()

    # Initialize the Hand tracking module
    with handsModule.Hands(
            static_image_mode=False,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=2) as hands:

        while True:
            # Read a frame from the camera
            ret, frame = capture.read()
            # black_frame = np.zeros_like(frame)

            if not ret:
                continue

            # Flip the frame horizontally for a mirrored view
            frame = cv2.flip(frame, 1)

            # Process the frame to detect hands
            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Create threads for hand calculations, hand drawing, and hand-inside check
            calc = threading.Thread(target=calculus,
                                    args=(results, finger_coordinates, distances, PROPORTIONAL_DISTANCES))
            draw = threading.Thread(target=drawFingers,
                                    args=(frame, finger_coordinates, circle_center, circle_radius, all_inside, on,
                                          PROPORTIONAL_DISTANCES, start_time))
            hand_inside_thread = threading.Thread(target=handInCircle, args=(results, all_inside, frame))

            if results.multi_hand_landmarks is not None:

                if not on:

                    # Start the hand-inside check thread
                    hand_inside_thread.start()
                    is_inside = True if all_inside.value == 1 else False

                    # Start timing if hand is inside the circle
                    start_time = time.time() if (
                            is_inside and start_time is None) else None if not is_inside else start_time

                    # Start hand drawing if not inside or if inside but not for more than 3 seconds
                    if not is_inside or (is_inside and time.time() - start_time <= 3):
                        draw.start()
                        draw.join()

                    # Turn on the control variable if hand is inside for more than 3 seconds
                    if is_inside and time.time() - start_time >= 3 or on:
                        on = True

                if on:
                    # Start hand calculations and hand drawing
                    calc.start()
                    calc.join()
                    draw.start()
                    draw.join()

            else:
                # Turn off the control variable and reset timer
                on = False
                start_time = 0.0

            # Display the frame with hand tracking information
            cv2.imshow('Test image', frame)

            # Break the loop when the 'Esc' key is pressed
            if cv2.waitKey(1) == 27:
                break

    # Close all OpenCV windows and release the camera
    cv2.destroyAllWindows()
    capture.release()


if __name__ == '__main__':
    Camera = threading.Thread(target=cam, args=(proportional_distances,))
    Camera.start()
    Arduino_process = threading.Thread(target=arduino_thread, args=(proportional_distances,))
    Arduino_process.start()
    Arduino_process.join()
