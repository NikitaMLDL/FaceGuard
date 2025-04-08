import numpy as np
import requests_mock
import sys
import os
import torch

# Добавляем путь к модели
sys.path.append("C:/Users/User/Desktop/ML_CV/airflow_classifier/Face_Net")

# Импортируем необходимые функции
from frontend.app import detect_face, is_user_in_db, add_user, delete_user

# Путь к модели YOLO
model_path = r"C:\Users\User\Desktop\ML_CV\airflow_classifier\Face_Net\frontend\yolov8n-face-lindevs.pt"

# Загружаем модель YOLO
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Модель не найдена по пути: {model_path}")

# Пример загрузки модели
model = torch.load(model_path)
model.eval()

# ============================
#        UNIT TESTS
# ============================

def test_detect_face():
    # Создаем пустое изображение (черный кадр)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Детектируем лицо на черном фоне
    face, x1, y1, x2, y2 = detect_face(frame, model)  # Передаем модель YOLO
    
    # Ожидаем, что лицо не будет найдено
    assert face is None
    assert x1 is None
    assert y1 is None
    assert x2 is None
    assert y2 is None

def test_is_user_in_db():
    # Создаем случайное изображение (100x100 пикселей)
    face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Мокаем HTTP-запросы
    with requests_mock.Mocker() as mock:
        mock.post("http://localhost:8000/check_user/", json={"message": "User exists in the system."})
        assert is_user_in_db(face) is True
        
        mock.post("http://localhost:8000/check_user/", json={"message": "User not found in the system."})
        assert is_user_in_db(face) is False

# ============================
#   INTEGRATION TESTS (API)
# ============================

def test_add_user():
    face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    with requests_mock.Mocker() as mock:
        mock.post("http://localhost:8000/add_face_embedding/", status_code=200)
        
        # Пытаемся добавить пользователя
        add_user(face)  # Не должно быть ошибок
        mock.assert_called_once_with("http://localhost:8000/add_face_embedding/")

def test_delete_user():
    face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    with requests_mock.Mocker() as mock:
        mock.post("http://localhost:8000/remove_face_embedding/", status_code=200)
        
        # Пытаемся удалить пользователя
        delete_user(face)  # Не должно быть ошибок
        mock.assert_called_once_with("http://localhost:8000/remove_face_embedding/")

# ============================
#    END-TO-END TEST (E2E)
# ============================

def test_end_to_end():
    """Проверяем весь процесс: детекция лица → проверка в БД → добавление → удаление"""
    face = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    with requests_mock.Mocker() as mock:
        # Мокаем API-ответы
        mock.post("http://localhost:8000/check_user/", json={"message": "User not found in the system."})
        mock.post("http://localhost:8000/add_face_embedding/", status_code=200)
        mock.post("http://localhost:8000/remove_face_embedding/", status_code=200)
        
        # Проверяем, что пользователя нет в базе данных
        assert is_user_in_db(face) is False
        
        # Добавляем пользователя
        add_user(face)
        
        # Проверяем, что пользователь добавлен (по вызову API)
        mock.post("http://localhost:8000/check_user/", json={"message": "User exists in the system."})
        assert is_user_in_db(face) is True
        
        # Теперь удаляем
        delete_user(face)
