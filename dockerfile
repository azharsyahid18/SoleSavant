# Gunakan image dasar dari Python
FROM python:3.9

# Set working directory
WORKDIR /app

# Salin semua file ke container
COPY . /app

# Instal dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ekspos port aplikasi
EXPOSE 8501

# Jalankan aplikasi
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
