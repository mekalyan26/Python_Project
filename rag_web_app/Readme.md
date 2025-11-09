## Run Application 
cd .\.venv\Scripts\
.\activate

### Backend 
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

#### Healthcheck
http://localhost:8000/health

### Frontend

 python -m http.server 5173 --bind 127.0.0.1

 # model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 1. Downloads/loads config.json
# 2. Sees it's a LlamaConfig
# 3. Uses LlamaForCausalLM class
# 4. Downloads/loads model weights
# 5. Returns initialized LlamaForCausalLM instance