# Образ для запуска Gradio-демо инференса.
FROM python:3.12-slim

WORKDIR /app

# Сначала зависимости — чтобы слои кешировались между сборками.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Затем исходники и обученный чекпойнт.
COPY . .

# Путь к чекпойнту можно переопределить при запуске: -e CHECKPOINT=...
ENV CHECKPOINT=checkpoints/linear_probe_best.pt
EXPOSE 7860

CMD ["python", "app/app.py"]
