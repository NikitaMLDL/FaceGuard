import faiss
import numpy as np
import os
import logging
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBService:
    """
    Class for working with a FAISS database without explicit IDs.
    """

    def __init__(self, index_file_path: str = "faiss_index.index", dim: int = 512):
        """
        Initializes the FAISS database.

        :param index_file_path: Path to the file for saving the index.
        :param dim: Dimensionality of the embeddings.
        """
        logger.info(f"Index size: {index_file_path}")
        self.index_file_path = index_file_path
        self.dim = dim
        self.index = None
        self.load_index()

    def load_index(self):
        """
        Loads the index from a file if it exists.
        If the index file does not exist, a new index is created.
        """
        if os.path.exists(self.index_file_path):
            try:
                self.index = faiss.read_index(self.index_file_path)
                logger.info(f"Index loaded from {self.index_file_path}")
            except Exception as e:
                logger.error(f"Failed to load index: {str(e)}")
                self.index = faiss.IndexFlatL2(self.dim)
        else:
            self.index = faiss.IndexFlatL2(self.dim)

    def save_index(self):
        """
        Saves the index to a file.
        """
        try:
            faiss.write_index(self.index, self.index_file_path)
            logger.info(f"Index saved to {self.index_file_path}")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")

    def add_embeddings(self, embeddings: np.ndarray):
        """
        Adds embeddings to FAISS.

        :param embeddings: Embedding vectors to be added.
        """
        try:
            embeddings = embeddings.astype(np.float32)
            self.index.add(embeddings)
            logger.info(f"Added {embeddings.shape[0]} embeddings to FAISS.")
            self.save_index()
        except Exception as e:
            logger.error(f"Error adding embeddings to FAISS: {str(e)}")

    def search_embedding(self, embedding: np.ndarray, k: int = 1, threshold: float = 0.6) -> List[int]:
        """
        Ищет наиболее похожие эмбеддинги в FAISS.

        :param embedding: Эмбеддинг, по которому выполняется поиск.
        :param k: Количество ближайших соседей.
        :param threshold: Пороговое значение дистанции (чем ниже, тем ближе).
        :return: Список индексов найденных эмбеддингов, удовлетворяющих порогу.
        """
        try:
            embedding = embedding.astype(np.float32).reshape(1, -1)
            distances, indices = self.index.search(embedding, k)

            # Фильтруем результаты по порогу расстояния
            valid_indices = [
                idx for dist, idx in zip(distances[0], indices[0]) if dist < threshold
            ]

            if valid_indices:
                logger.info(f"Found {len(valid_indices)} nearest neighbors: {valid_indices}")
                return valid_indices
            else:
                logger.info("No embeddings found within the threshold.")
                return [-1]  # Если ни один эмбеддинг не прошел порог, возвращаем -1

        except Exception as e:
            logger.error(f"Error searching in FAISS: {str(e)}")
            return []


    def remove_embedding_by_embedding(self, embedding: np.ndarray):
        """
        Удаляет эмбеддинг из FAISS, если он существует.

        :param embedding: Вектор эмбеддинга, который нужно удалить.
        """
        try:
            if self.index.ntotal == 0:
                logger.error("FAISS index is empty. Nothing to remove.")
                return

            # Преобразуем эмбеддинг в нужный формат
            embedding = embedding.astype(np.float32).reshape(1, -1)

            # Ищем ближайший эмбеддинг в FAISS
            _, indices = self.index.search(embedding, 1)  # Ищем 1 ближайший вектор
            idx = indices[0][0]

            # Проверяем, найден ли в базе эмбеддинг
            if idx == -1:
                logger.error("Embedding not found in FAISS. No changes made.")
                return

            # Получаем все вектора и удаляем нужный
            embeddings = self.index.reconstruct_n(0, self.index.ntotal)
            embeddings = np.delete(embeddings, idx, axis=0)

            # Пересоздаем индекс и загружаем обновленные вектора
            self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(embeddings)

            logger.info(f"Removed embedding at index {idx}.")
            self.save_index()
        
        except Exception as e:
            logger.error(f"Error removing embedding: {str(e)}")