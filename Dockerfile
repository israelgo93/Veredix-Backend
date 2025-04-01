# Usa la imagen oficial de Python
FROM python:3.10-slim

# Actualizar repositorios e instalar git
RUN apt-get update && apt-get install -y git

# Crea el directorio de trabajo
WORKDIR /app

# Copia requirements e instálalos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de archivos de tu proyecto al contenedor
COPY . .

# Exponemos el puerto 7777 para FastAPI
EXPOSE 7777

# Comando de arranque
CMD ["python", "playground.py"]
