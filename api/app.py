from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.preprocessor import TextPreprocessor
import joblib

app = FastAPI()

# 设置模板目录
templates = Jinja2Templates(directory="api/templates")

# 加载预处理器和模型
preprocessor = TextPreprocessor()
model_path = os.path.join("data", "models", "best_model.joblib")
vectorizer_path = os.path.join("data", "models", "vectorizer.joblib")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(text: str = Form(...)):
    # 预处理文本
    processed_text = preprocessor.preprocess(text)
    # 使用向量器转换文本
    text_vectorized = vectorizer.transform([processed_text])
    
    # 进行预测
    prediction = model.predict(text_vectorized)[0]
    
    # 返回JSON响应
    return JSONResponse(content={"category": prediction})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 