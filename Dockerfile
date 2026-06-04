# Образ для запуска Gradio-демо инференса.
FROM python:3.12-slim

WORKDIR /app

# Сначала зависимости — чтобы слои кешировались между сборками.
COPY requirements.txt .
RUN pip install --no-cache-dir \
        torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Затем исходники (веса в образ не кладём — они качаются с Hugging Face Hub).
COPY . .

# Веса берутся из этого репозитория на Hub при первом запуске, если локально их
# нет. Переопределить можно при запуске: -e HF_REPO_ID=... или -e CHECKPOINT=...
ENV HF_REPO_ID=A11Sunday/vit-cat-dog-panda
ENV CHECKPOINT=checkpoints/linear_probe_best.pt
EXPOSE 7860

CMD ["python", "app/app.py"]
