FROM python:3.14-slim
WORKDIR /app

COPY requirements.txt .
# CPU-only torch: the default CUDA wheels add several GB
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY src/ src/
COPY download_data.py app.py ./
# Bake the dataset, FAISS index and model weights into the image so cold starts don't re-embed
RUN python download_data.py \
 && python -c "from src.pipeline import build_retriever; build_retriever()" \
 && rm -f data/*.zip data/*.tar.gz
ENV HF_HUB_OFFLINE=1 PORT=8080

# LLM_API_KEY comes from Secret Manager at deploy time, never baked in
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port $PORT"]
