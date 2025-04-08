from fastapi import APIRouter, HTTPException, File, UploadFile, Request
import numpy as np
import logging
from .faiss_service import VectorDBService
import os
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from io import BytesIO
from fastapi.responses import JSONResponse
import logging
import torch


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
router = APIRouter()
# Инициализируем FAISS индекс (это необходимо настроить для работы)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создаем объект базы данных пользователей
user_db = VectorDBService()
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='casia-webface').eval()


# Функция для извлечения эмбеддинга из изображения с помощью InceptionResnetV1
def get_embedding_from_image(image_bytes: bytes) -> np.ndarray:
    """
    Преобразует изображение в эмбеддинг с использованием InceptionResnetV1.
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)

    # Детекция лица и обрезка
    face = mtcnn(image)
    if face is None:
        raise HTTPException(status_code=404, detail="No face detected in the image.")

    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0))  # Добавляем batch dimension перед подачей в сеть

    return embedding[0].cpu().numpy()  # Возвращаем NumPy массив


@router.post("/check_user/")
async def check_user_in_system(request: Request, file: UploadFile = File(...)):
    """
    Эндпоинт для проверки, есть ли пользователь в системе по изображению.
    """
    # Читаем изображение
    image_bytes = await file.read()

    # try:
    embedding = get_embedding_from_image(image_bytes)
    logger.info(embedding)
    # except HTTPException as e:
    #     logger.info('Failed')
    #     raise e  # Просто передаем ошибку дальше

    # Проверяем, есть ли пользователь в базе данных
    search_result = user_db.search_embedding(embedding)

    logger.info(f"Search result: {search_result}")

    if search_result != [-1]:
        return {"message": "User exists in the system."}
    else:
        return {"message": "User not found in the system."}



@router.post("/add_face_embedding/")
async def add_face_embedding(request: Request, file: UploadFile = File(...)):
    try:
        # Получаем изображение из запроса
        image_bytes = await file.read()

        # Получаем эмбеддинг из изображения
        embedding = get_embedding_from_image(image_bytes)

        # Добавляем эмбеддинг в базу данных (например, в FAISS)
        user_db.add_embeddings(embedding.reshape(1, -1))  # Добавляем эмбеддинг как 2D массив (1, -1)

        return JSONResponse(content={"message": "Embedding added successfully!"}, status_code=200)

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return JSONResponse(content={"message": f"Error: {str(e)}"}, status_code=400)


@router.post("/remove_face_embedding/")
async def remove_face_embedding(request: Request, file: UploadFile = File(...)):
    """
    Удаляет эмбеддинг пользователя из базы данных по загруженному изображению.
    """
    try:
        # Читаем изображение
        image_bytes = await file.read()

        # Получаем эмбеддинг из изображения
        embedding = get_embedding_from_image(image_bytes)

        # Преобразуем эмбеддинг в нужный формат
        embedding = np.array(embedding).astype(np.float32)

        # Удаляем эмбеддинг из базы
        user_db.remove_embedding_by_embedding(embedding)

        return {"message": "Embedding removed successfully!"}
    
    except HTTPException as e:
        raise e  # Если ошибка связана с обработкой изображения, пробрасываем её дальше
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")