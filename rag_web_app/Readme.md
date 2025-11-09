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

 