import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from predictor import MalariaPredictor

# Initialize the predictor
predictor = MalariaPredictor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure model is loaded on startup
    predictor.load_model()
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = predictor.predict(contents)
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return result

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Mount static files
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
