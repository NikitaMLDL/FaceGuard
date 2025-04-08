import streamlit as st
import cv2
from ultralytics import YOLO
import requests
from PIL import Image
import io
import logging

# Загружаем модель YOLOv8 для детекции лиц
yolo_net = YOLO("yolov8n-face-lindevs.pt")  # Укажите путь к вашему файлу модели
API_BASE_URL = "http://localhost:8000"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Функция для обработки видео потока и нахождения лица
def detect_face(frame):
    # Преобразуем изображение в формат RGB для YOLO
    results = yolo_net(frame)  # Получаем результат детекции
    
    # Извлекаем найденные лица
    faces = results[0].boxes.xyxy
    
    # Если лица не найдены, возвращаем None
    if len(faces) == 0:
        return None, None, None, None, None

    # Извлекаем координаты первого лица (x1, y1, x2, y2)
    x1, y1, x2, y2 = faces[0].cpu().numpy()
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Обрезаем изображение по координатам (с учетом округления)
    face = frame[y1:y2, x1:x2]

    # Изменяем размер изображения лица для дальнейшей обработки (например, для сохранения)
    face = cv2.resize(face, (100, 100))

    return face, x1, y1, x2, y2


# Заглушка для проверки, есть ли пользователь в базе данных
def is_user_in_db(face):
    try:
        pil_image = Image.fromarray(face)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        files = {'file': ('image.png', img_byte_arr, 'image/png')}
        res = requests.post(f"{API_BASE_URL}/check_user/", files=files)
        # res.raise_for_status()

        response_json = res.json()  # Получаем JSON
        return response_json.get('message') == "User exists in the system."
    except Exception as e:
        logger.error(f"Error in is_user_in_db: {e}")
        return False


# Заглушка для удаления пользователя
def delete_user(face):
    """
    Отправляет изображение пользователя в API для удаления эмбеддинга.
    """

    # Конвертируем изображение в байты
    pil_image = Image.fromarray(face)
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Отправляем изображение на сервер
    files = {'file': ('image.png', img_byte_arr, 'image/png')}

    try:
        res = requests.post(f"{API_BASE_URL}/remove_face_embedding/", files=files)
        res.raise_for_status()

        # Показываем сообщение об успехе
        st.success("User deleted from the database!")
    except requests.exceptions.RequestException as e:
        st.error(f"Error deleting user: {e}")

# Заглушка для добавления пользователя
def add_user(face):
    # Конвертируем изображение в байты
    pil_image = Image.fromarray(face)
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Отправляем изображение на сервер
    files = {'file': ('image.png', img_byte_arr, 'image/png')}
    
    try:
        res = requests.post(f"{API_BASE_URL}/add_face_embedding/", files=files)
        res.raise_for_status()
        
        # Показываем сообщение об успехе
        st.success("User added to the database!")
    except requests.exceptions.RequestException as e:
        st.error(f"Error adding user: {e}")

# Основной Streamlit код
st.title("Face Recognition using YOLOv8")

# Инициализируем состояние сессии
if "snapshot_taken" not in st.session_state:
    st.session_state.snapshot_taken = False

# Открываем видеопоток с камеры
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
stframe = st.empty()  # Пустой контейнер для видео потока
take_snapshot = st.button("Take Snapshot")  # Кнопка для сохранения снимка

# Создаем пустое место для сообщения
status_message = st.empty()

# Обработаем нажатие кнопки "Take Snapshot"
if take_snapshot:
    st.session_state.snapshot_taken = True

# Основной цикл видео потока
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Детекция лица на видео
    face, x1, y1, x2, y2 = detect_face(frame)

    # Если лицо найдено, рисуем квадрат
    if face is not None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Отображаем видео с детекцией
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    stframe.image(frame_rgb, channels="RGB")

    # Если снимок был сделан, показываем его
    if st.session_state.snapshot_taken and face is not None:
        # Проверяем, есть ли пользователь в базе данных
        if is_user_in_db(face):
            st.success("User found in the system.")
            delete_button = st.button("Delete User")
            if delete_button:
                delete_user(face)  # Удаляем пользователя из базы данных
                st.session_state.snapshot_taken = False
                st.rerun()  # Перезагружаем страницу после удаления
        else:
            st.warning("User not found in the system.")
            add_button = st.button("Add User")
            if add_button:
                add_user(face)  # Добавляем пользователя в базу данных
                st.session_state.snapshot_taken = False
                status_message.empty()
                st.rerun()  # Перезагружаем страницу после добавления

        # Появляется кнопка продолжить только если снимок был сделан
        continue_button = st.button("Continue")
        
        # Если кнопка "Continue" нажата
        if continue_button:
            # После нажатия "Continue" перезагружаем страницу
            st.session_state.snapshot_taken = False
            status_message.empty()  # Удаляем сообщение об успехе
            st.rerun()

        break  # Прерываем цикл, чтобы завершить обработку после снимка

# Закрываем видеопоток после завершения
cap.release()
cv2.destroyAllWindows()
