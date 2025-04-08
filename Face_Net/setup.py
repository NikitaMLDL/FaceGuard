from setuptools import setup, find_packages

def read_requirements(file_path):
    """
    Функция для чтения зависимостей из файла requirements.txt
    """
    with open(file_path) as f:
        return [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="faiss_db",  # Название пакета
    version="1.0.0",
    packages=find_packages(where='backend/app/db'),  # Указываем, где искать пакеты внутри backend/app
    install_requires=read_requirements("backend/requirements.txt"),  # Зависимости для установки
    author="Nikita Abramov",
    author_email="nikitaabr83@gmail.com",
    description="Face recognition access control system",  # Описание проекта
    keywords="face recognition, access control, security",  # Ключевые слова
    python_requires=">=3.12"  # Минимальная версия Python
)
