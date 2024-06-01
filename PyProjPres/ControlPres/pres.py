import argparse
import colorsys
from contextlib import ExitStack
import csv
import time  # Добавьте этот импорт в начало вашего кода
import numpy as np
import keyboard
import mediapipe as mp
import cv2

from datetime import datetime
from math import atan, atan2, pi, degrees
from numpy import concatenate
from scipy.spatial import distance as dist

# Задаем стандартный стиль для отображения ключевых точек тела
DEFAULT_LANDMARKS_STYLE = mp.solutions.drawing_styles.get_default_pose_landmarks_style()

# Опционально записываем видеопоток в AVI файл с временной меткой в текущем каталоге
RECORDING_FILENAME = str(datetime.now()).replace('.','').replace(':','') + '.avi'
FPS = 10

# Порог видимости ключевых точек тела
VISIBILITY_THRESHOLD = .8

# Маржа для определения, вытянута ли конечность (в градусах)
STRAIGHT_LIMB_MARGIN = 20
person_detected = False

# Длина нижней конечности в виде доли от верхней конечности
EXTENDED_LIMB_MARGIN = .8

# Максимальное расстояние от запястья до противоположного локтя, относительно ширины рта
ARM_CROSSED_RATIO = 2

# Порог расстояния для проверки, находятся ли руки перед ртом (в единицах от 0 до 1)
MOUTH_COVER_THRESHOLD = .03

# Минимальный угол поднимания ноги от горизонтали (в градусах)
LEG_LIFT_MIN = -10

# Порог для определения приседания (в единицах от 0 до 1)
SQUAT_THRESHOLD = .1

# Порог для определения прыжка (разница высоты бедер в кадре)
JUMP_THRESHOLD = .0001

# Словарь семафоров для различных жестов и действий
semaphores = {}

# Угол расширения ноги (в градусах от вертикального стояния); должен быть делителем 90
LEG_EXTEND_ANGLE = 18

# Словарь углов расширения ног для определения ноги и ее положения
leg_extension_angles = {
  (-90, -90 + LEG_EXTEND_ANGLE): (True, 0),  # правая нога, низкая
  (-90, -90 + 2*LEG_EXTEND_ANGLE): (True, 1),  # правая нога, высокая
  (270 - LEG_EXTEND_ANGLE, -90): (False, 0),  # левая нога, низкая
  (270 - 2*LEG_EXTEND_ANGLE, -90): (False, 1),  # левая нога, высокая
}

# Количество последних кадров, используемых для анализа движений
FRAME_HISTORY = 8

# Половина истории кадров
HALF_HISTORY = int(FRAME_HISTORY/2)

# Пустой кадр для заполнения истории
empty_frame = {
  'hipL_y': 0,
  'hipR_y': 0,
  'hips_dy': 0,
}

# История последних кадров для каждого спикера
last_frames = FRAME_HISTORY*[empty_frame.copy()]

# Середина кадра
frame_midpoint = (0,0)

# Последние нажатые клавиши для каждого спикера
last_keys = [[]]

# Функция для загрузки карты клавиш из CSV файла
def map_keys(file_name, player_count):
  global semaphores

  with open('carts/' + (file_name or 'RL.csv')) as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader) # Пропускаем первую строку (заголовок)
    for player_cardinal, part, position, keys, repeat, action_name in csv_reader:
      player_index = player_count - int(player_cardinal) # Реверс и индексация с 0
      semaphores[(player_index, part, int(position))] = {
        'keys': keys.split(' '),
        'name': action_name,
        'repeat': bool(int(repeat)),
      }
    print("Successfully read in:", semaphores)

# Функция для вычисления угла между тремя точками
def get_angle(a, b, c):
  ang = degrees(atan2(c['y']-b['y'], c['x']-b['x']) - atan2(a['y']-b['y'], a['x']-b['x']))
  return ang + 360 if ang < 0 else ang

# Функция для проверки, отсутствует ли видимая часть ключевой точки
def is_missing(part):
  return any(joint['visibility'] < VISIBILITY_THRESHOLD for joint in part)

# Функция для проверки, указывает ли конечность в прямом направлении
def is_limb_pointing(upper, mid, lower):
  if is_missing([upper, mid, lower]):
    return False
  limb_angle = get_angle(upper, mid, lower)
  is_in_line = abs(180 - limb_angle) < STRAIGHT_LIMB_MARGIN
  if is_in_line:
    upper_length = dist.euclidean([upper['x'], upper['y']], [mid['x'], mid['y']])
    lower_length = dist.euclidean([lower['x'], lower['y']], [mid['x'], mid['y']])
    is_extended = lower_length > EXTENDED_LIMB_MARGIN * upper_length
    return is_extended
  return False


# Функция для получения направления конечности
def get_limb_direction(arm, closest_degrees=45):
  # Рассчитываем угол между запястьем и плечом
  dy = arm[2]['y'] - arm[0]['y']  # запястье -> плечо
  dx = arm[2]['x'] - arm[0]['x']
  angle = degrees(atan(dy / dx))

  # Корректируем угол для отрицательных значений (левая сторона)
  if dx < 0:
    angle += 180

  # Округляем угол до ближайшего closest_degrees; 45 для семафора
  mod_close = angle % closest_degrees
  angle -= mod_close
  if mod_close > closest_degrees / 2:
    angle += closest_degrees

  angle = int(angle)

  # Преобразуем 270 градусов в -90 градусов для удобства интерпретации
  if angle == 270:
    angle = -90

  return angle


# Функция для проверки, пересекаются ли руки
def is_arms_crossed(elbowL, wristL, elbowR, wristR, mouth_width):
  # Максимальное расстояние, на котором руки считаются пересекающимися
  max_dist = mouth_width * ARM_CROSSED_RATIO

  # Проверка пересечения левой руки с правой и наоборот
  return is_arm_crossed(elbowL, wristR, max_dist) and is_arm_crossed(elbowR, wristL, max_dist)


# Функция для проверки, пересекается ли одна конечность с другой
def is_arm_crossed(elbow, wrist, max_dist):
  # Проверка, находится ли запястье на достаточном расстоянии от локтя
  return dist.euclidean([elbow['x'], elbow['y']], [wrist['x'], wrist['y']]) < max_dist


# Функция для проверки, поднята ли нога
def is_leg_lifted(leg):
  if is_missing(leg):
    return False

  # Рассчитываем угол между коленом и бедром
  dy = leg[1]['y'] - leg[0]['y']  # колено -> бедро
  dx = leg[1]['x'] - leg[0]['x']
  angle = degrees(atan2(dy, dx))

  # Проверка, поднята ли нога
  return angle > LEG_LIFT_MIN


# Функция для проверки, выполняется ли прыжок
def is_jumping(i, hipL, hipR):
    global last_frames

    if is_missing([hipL, hipR]):
        return False

    # Запоминание текущей высоты бедра для последующего сравнения
    last_frames[i][-1]['hipL_y'] = hipL['y']
    last_frames[i][-1]['hipR_y'] = hipR['y']

    # Проверка, поднимаются или опускаются бедра по сравнению с предыдущими кадрами
    if (hipL['y'] > last_frames[i][-2]['hipL_y'] + JUMP_THRESHOLD) and (
            hipR['y'] > last_frames[i][-2]['hipR_y'] + JUMP_THRESHOLD):
        last_frames[i][-1]['hips_dy'] = 1  # подъем
    elif (hipL['y'] < last_frames[i][-2]['hipL_y'] - JUMP_THRESHOLD) and (
            hipR['y'] < last_frames[i][-2]['hipR_y'] - JUMP_THRESHOLD):
        last_frames[i][-1]['hips_dy'] = -1  # спуск
    else:
        last_frames[i][-1]['hips_dy'] = 0  # незначительное изменение высоты

    # Проверка выполнения прыжка (подъем в первой половине и спуск во второй половине истории)
    jump_up = all(frame['hips_dy'] == 1 for frame in last_frames[i][:HALF_HISTORY])
    get_down = all(frame['hips_dy'] == -1 for frame in last_frames[i][HALF_HISTORY:])
    return jump_up and get_down


# Функция для проверки, закрыт ли рот обоими ладонями
def is_mouth_covered(mouth, palms):
    if is_missing(palms):
        return False

    # Расчет разницы в координатах между ртом и ладонями
    dxL = (mouth[0]['x'] - palms[0]['x'])
    dyL = (mouth[0]['y'] - palms[0]['y'])
    dxR = (mouth[1]['x'] - palms[1]['x'])
    dyR = (mouth[1]['y'] - palms[1]['y'])

    # Проверка, находится ли рот между ладонями
    return all(abs(d) < MOUTH_COVER_THRESHOLD for d in [dxL, dyL, dxR, dyR])


# Функция для проверки, выполняется ли приседание
def is_squatting(hipL, kneeL, hipR, kneeR):
    if is_missing([hipL, kneeL, hipR, kneeR]):
        return False

    # Расчет вертикального расстояния между бедром и коленом
    dyL = abs(hipL['y'] - kneeL['y'])
    dyR = abs(hipR['y'] - kneeR['y'])

    # Проверка, находятся ли бедра и колени на достаточном расстоянии друг от друга
    return (dyL < SQUAT_THRESHOLD) and (dyR < SQUAT_THRESHOLD)

def match_and_type(player_num, parts_and_actions, image, display_only):
    global semaphores, last_keys

    new_keys = []
    new_keys_to_repeat = []

    for (part_or_action, position) in parts_and_actions:
        match = semaphores.get((player_num, part_or_action, position), '')
        if match:
            if match.get('repeat'):
                new_keys_to_repeat += [match.get('keys', '')]
            else:
                new_keys += [match.get('keys', '')]

    all_new_keys = new_keys + new_keys_to_repeat

    for hotkey in last_keys[player_num]:
        if (hotkey not in all_new_keys):
            print("releasing:", hotkey)
            keyboard.release(hotkey)

    if all_new_keys and all_new_keys != last_keys[player_num]:
        output(all_new_keys, last_keys[player_num], False, image, display_only)

    last_keys[player_num] = all_new_keys


def output(keys, previous_keys, repeat, image, display_only):
    for hotkey in keys:
        keystring = '+'.join(key for key in hotkey if key not in previous_keys)
        if len(keystring):
            if not repeat and keystring not in previous_keys:
                if display_only:
                    cv2.putText(image, keystring, frame_midpoint,
                                cv2.FONT_HERSHEY_SIMPLEX, 20, (0, 0, 255), 20)
                else:
                    print("pressing:", keystring)
                    keyboard.press(keystring)
                    time.sleep(1)  # Добавленная задержка в 3 секунды
            elif repeat:
                print("REPEAT: press & release", keystring)
                keyboard.press_and_release(keystring)


# Функция для отображения изображения и, при необходимости, завершения программы
def render_and_maybe_exit(image, recording):

  cv2.imshow('KIP-CV-PRES', image)
  if recording:
    recording.write(image)
  return cv2.waitKey(5) & 0xFF == 27

def draw_line(image, point1, point2, color):
    cv2.line(image, point1, point2, color, 6)

# Функция для обработки поз
def process_poses(image, pose_models, draw_landmarks, flip, display_only):
    global last_frames, frame_midpoint, last_keys, person_detected
    frame_size = (int(image.shape[1]), int(image.shape[0]))
    mp_drawing = mp.solutions.drawing_utils

    h, w, c = image.shape
    mp_holistic = mp.solutions.holistic

    # Преобразование изображения в цветовую схему RGB для mediapipe
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Разделение изображения на несколько частей в зависимости от числа pose_models
    width = image.shape[1]
    splits = len(pose_models)
    split_len = width // splits
    images = [image[:, i:i + split_len] for i in range(0, width, split_len)]

    # Отрисовка разделительных линий на изображении
    for mark in range(0, width, split_len):
        cv2.line(image, (mark, 0), (mark, width), (255, 255, 255), 1)

    # Обработка поз для каждой части изображения
    pose_results = [pose_models[i].process(images[i]) for i in range(0, splits)]

    # Отрисовка landmarks, если необходимо
    if draw_landmarks:
        for i, image in enumerate(images):
            mp_drawing.draw_landmarks(
                image,
                pose_results[i].pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                DEFAULT_LANDMARKS_STYLE)

    # Объединение изображений обратно в одно
    image = concatenate(images, axis=1)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Отражение изображения, если установлен параметр flip
    if flip:
        image = cv2.flip(image, 1)

    # Обработка действий на основе поз для каждого игрока
    for player_num, pose_result in enumerate(pose_results):
        actions = []
        if not person_detected:

            if pose_result.pose_landmarks:
                # Извлечение координат ключевых точек для определения прямоугольника вокруг всего тела
                landmarks = pose_result.pose_landmarks.landmark
                xs = [int(point.x * frame_size[0]) for point in landmarks]
                ys = [int(point.y * frame_size[1]) for point in landmarks]

                # Определение прямоугольника, охватывающего всю область тела
                body_rect = (
                    min(xs), min(ys),  # левая верхняя точка
                    max(xs), max(ys)  # правая нижняя точка
                )

                # Рисование зеленого прямоугольника вокруг всего тела
                cv2.rectangle(image, (body_rect[0], body_rect[1]), (body_rect[2], body_rect[3]), (0, 255, 0), 3)
                cv2.putText(image, "СПИКЕР", (body_rect[0] - 40, body_rect[1]-20),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (37, 156, 0), 2)

                mp_drawing.draw_landmarks(image, pose_result.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 225, 0), thickness=2,
                                                                                       circle_radius=1),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 246, 117),
                                                                                         thickness=2))

                # Подготовка для хранения последних кадров движения в течение некоторого времени
                last_frames[player_num] = last_frames[player_num][1:] + [empty_frame.copy()]

                body = []
                # Преобразование координат landmarks в относительные координаты
                right_hand_coords = None
                left_hand_coords = None

                # Преобразование координат landmarks в относительные координаты
                for point in pose_result.pose_landmarks.landmark:
                    body.append({
                        'x': 1 - point.x,
                        'y': 1 - point.y,
                        'visibility': point.visibility
                    })
                    if point == pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]:
                        right_hand_coords = (int(point.x * frame_size[0]), int(point.y * frame_size[1]))

                    if point == pose_result.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]:
                        left_hand_coords = (int(point.x * frame_size[0]), int(point.y * frame_size[1]))

                # После обработки всех ключевых точек для каждого спикера
                if right_hand_coords:
                    cv2.circle(image, (right_hand_coords[0], right_hand_coords[1]), 50, (255, 0, 0), 3) # Синие контуры

                    cv2.putText(image, "Правая", (right_hand_coords[0] - 40, right_hand_coords[1] - 60),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (37, 156, 0), 2)

                    point_11 = (int(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x * w),
                                int(landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y * h))
                    point_13 = (int(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW].x * w),
                                int(landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW].y * h))
                    point_15 = (int(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST].x * w),
                                int(landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST].y * h))

                    draw_line(image, point_11, point_13, (242, 90, 92))  # Линия от плеча до точки 12 на правой руке
                    draw_line(image, point_11, point_13,
                              (242, 90, 92))  # Линия от точки 12 до точки 14 на правой руке
                    draw_line(image, point_13, point_15,
                              (242, 90, 92))  # Линия от точки 14 до точки 16 на правой руке

                if left_hand_coords:
                    cv2.circle(image, (left_hand_coords[0], left_hand_coords[1]), 50, (0, 0, 255), 2) # Красные контуры

                    cv2.putText(image, "Левая", (left_hand_coords[0] - 40, left_hand_coords[1] - 60),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (37, 156, 0), 2)

                    point_12 = (int(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].x * w),
                                int(landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER].y * h))
                    point_14 = (int(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].x * w),
                                int(landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW].y * h))
                    point_16 = (int(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].x * w),
                                int(landmarks[mp_holistic.PoseLandmark.LEFT_WRIST].y * h))

                    draw_line(image, point_12, point_14, (120, 120, 255))  # Линия от плеча до точки 12 на правой руке
                    draw_line(image, point_12, point_14, (120, 120, 255))  # Линия от точки 12 до точки 14 на правой руке
                    draw_line(image, point_14, point_16, (120, 120, 255))  # Линия от точки 14 до точки 16 на правой руке

                kneeL, kneeR = body[25], body[26]
                hipL, hipR = body[23], body[24]
                legL = (hipL, kneeL, body[27])  # + ankle
                legR = (hipR, kneeR, body[28])  # + ankle

                if is_squatting(hipL, kneeL, hipR, kneeR):
                    # squat (hips <> knees ~horizontal)
                    actions += [('squat', 1)]
                elif is_leg_lifted(legL):  # одна нога поднята ~горизонтально
                    actions += [('left leg', 2)]
                elif is_leg_lifted(legR):
                    actions += [('right leg', 2)]
                else:
                    # углы разогнутости ног
                    if is_limb_pointing(*legL) and is_limb_pointing(*legR):
                        legL_angle = get_limb_direction(legL, LEG_EXTEND_ANGLE)
                        legR_angle = get_limb_direction(legR, LEG_EXTEND_ANGLE)
                        is_right, is_high = leg_extension_angles.get((legL_angle, legR_angle), (None, None))
                        if is_high is not None:
                            which_leg = ('right' if is_right else 'left') + ' leg'
                            actions += [(which_leg, is_high)]

                # jump (подъем и опускание бедер)
                if is_jumping(player_num, hipL, hipR):
                    actions += [('jump', 1)]

                # рот закрыт обеими ладонями
                mouth = (body[9], body[10])
                palms = (body[19], body[20])
                if is_mouth_covered(mouth, palms):
                    actions += [('mouth', 1)]

                # руки перекрещены: запястья близки к противоположным локтям
                shoulderL, elbowL, wristL = body[11], body[13], body[15]
                armL = (shoulderL, elbowL, wristL)
                shoulderR, elbowR, wristR = body[12], body[14], body[16]
                armR = (shoulderR, elbowR, wristR)
                mouth_width = abs(mouth[1]['x'] - mouth[0]['x'])
                if is_arms_crossed(elbowL, wristL, elbowR, wristR, mouth_width):
                    actions += [('crossed arms', 1)]

                # углы разогнутости одной руки
                for (arm, is_right) in [(armL, False), (armR, True)]:
                    if is_limb_pointing(*arm):
                        arm_angle = get_limb_direction(arm)
                        which_arm = ('right' if is_right else 'left') + ' arm'
                        actions += [(which_arm, arm_angle)]

        # Обработка действий и нажатия клавиш
        if actions or last_keys[player_num]:
            match_and_type(player_num, actions, image, display_only)

    return image

def draw_lines(image, landmarks, connections, colors):
    import random
    num_particles = 10
    particle_radius = 3
    height, width, _ = image.shape
    landmark_list = landmarks.landmark
    landmark_coords = [(int(landmark.x * width), int(landmark.y * height)) for landmark in landmark_list]

    for i, connection in enumerate(connections):
        start_idx, end_idx = connection
        start_point = landmark_coords[start_idx]
        end_point = landmark_coords[end_idx]
        color = colors[i]



        # Calculate the direction vector of the line
        direction_vector = np.array(end_point) - np.array(start_point)

        # Calculate the step size for placing particles along the line
        step_size = 1.0 / (num_particles + 1)

        # Place particles along the line
        for j in range(1, num_particles + 1):
            # Calculate the position of the particle along the line
            particle_position = (start_point[0] + j * step_size * direction_vector[0],
                                 start_point[1] + j * step_size * direction_vector[1])

            # Add random jitter to the particle position for a more natural look
            particle_position = (int(particle_position[0] + random.randint(-3, 3)),
                                 int(particle_position[1] + random.randint(-3, 3)))

            # Get hue for particle color based on its position
            hue = (particle_position[0] + particle_position[1]) % 360
            particle_color = tuple(int(255 * i) for i in colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0))

            # Draw particle
            cv2.circle(image, particle_position, particle_radius, particle_color, -1)

    # Draw landmarks
    for landmark in landmark_coords:
        hue = (landmark[0] + landmark[1]) % 360
        color = tuple(int(255 * i) for i in colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0))
        cv2.circle(image, landmark, 5, color, -1)

# главная
def main():
  global last_frames, last_keys, frame_midpoint

  indexCam = int(input('Введите индекс камеры: '))


  # Создание двух окон
  cv2.namedWindow('Black Pose', cv2.WINDOW_NORMAL)

  # Парсинг аргументов командной строки
  parser = argparse.ArgumentParser()

  # Парсинг аргументов командной строки
  parser = argparse.ArgumentParser()
  parser.add_argument('--map', '-m', help='File to import for mapped keys')
  parser.add_argument('--input', '-i', help='Input video device or file (number or path), defaults to 0', default='0')
  parser.add_argument('--flip', '-f', help='Set to any value to flip resulting output (selfie view)')
  parser.add_argument('--landmarks', '-l', help='Set to any value to draw body landmarks')
  parser.add_argument('--record', '-r', help='Set to any value to save a timestamped AVI in the current directory')
  parser.add_argument('--display', '-d', help='Set to any value to only visually display output rather than type')
  parser.add_argument('--split', '-s', help='Split the screen into a positive integer of separate regions, defaults to 1', default='1')
  args = parser.parse_args()

  INPUT = indexCam#int(args.input) if args.input.isdigit() else args.input
  FLIP = args.flip is not None
  DRAW_LANDMARKS = args.landmarks is not None
  RECORDING = args.record is not None
  DISPLAY_ONLY = args.display is not None
  SPLIT = int(args.split)

  # Инициализация переменных для каждого "раздела" изображения
  last_frames = SPLIT * [last_frames.copy()]
  last_keys = SPLIT * [[]]

  # Захват видео
  cap = cv2.VideoCapture(INPUT)

  # Получение размера кадра и середины кадра
  frame_size = (int(cap.get(3)), int(cap.get(4)))
  frame_midpoint = (int(frame_size[0] / 2), int(frame_size[1] / 2))

  # Инициализация записи видео, если указано
  recording = cv2.VideoWriter(RECORDING_FILENAME,
                              cv2.VideoWriter_fourcc(*'MJPG'), FPS, frame_size) if RECORDING else None

  # Загрузка карты клавиш для каждого "раздела"
  MAP_FILE = args.map
  map_keys(MAP_FILE, SPLIT)

  mp_pose = mp.solutions.pose
  # Инициализация модели
  pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

  # Инициализация mediapipe.pose для каждого "раздела"
  with ExitStack() as stack:
    pose_models = SPLIT * [stack.enter_context(mp.solutions.pose.Pose())]

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break


        results = pose.process(image)
        black_image = np.zeros_like(image)

        if results.pose_landmarks is not None:
            hue_values = np.linspace(0, 360, len(mp_pose.POSE_CONNECTIONS) + 1)[:-1]
            colors = [tuple(int(255 * i) for i in colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)) for hue in hue_values]

            draw_lines(black_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, colors)
            cv2.resizeWindow('Black Pose', 450, 300)
            cv2.setWindowProperty("Black Pose", cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow('Black Pose', black_image)

        # Обработка поз и отображение результата в главном окне
        image_main = process_poses(image, pose_models, DRAW_LANDMARKS, FLIP, DISPLAY_ONLY)

        # Проверка наличия команды завершения от пользователя
        if render_and_maybe_exit(image_main, recording):
            break

  # Завершение записи видео
  if RECORDING:
    recording.release()

  # Освобождение видеозахвата и закрытие окон OpenCV
  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  main()

