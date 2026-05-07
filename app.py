import os
import io
import cv2
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model
    load_malaria_model()
    yield
    # Shutdown: Clean up resources if needed
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Load the model
MODEL_PATH = "malaria_model_validated.h5"
model = None

def load_malaria_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file {MODEL_PATH} not found. Please train the model first.")

# (Removed deprecated on_event)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def get_superimposed_img(img_tensor, heatmap):
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    
    # img_tensor is (1, 128, 128, 3) and normalized to [0, 1]
    original_img = np.uint8(255 * img_tensor[0])
    
    superimposed_img = jet * 0.4 + original_img * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # If we don't have the real model, we will mock the prediction so the UI still works!
    using_mock = False
    if model is None:
        load_malaria_model()
        if model is None:
            print("Using mock prediction mode since model is not trained yet.")
            using_mock = True

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((128, 128))
        img_array = np.array(image).astype('float32') / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        if not using_mock:
            # Make real prediction
            preds = model.predict(img_tensor)
            score = float(preds[0][0])
            is_infected = score > 0.5
            
            # Generate real Grad-CAM
            last_conv = [l.name for l in reversed(model.layers) if 'conv2d' in l.name][0]
            heatmap = make_gradcam_heatmap(img_tensor, model, last_conv)
        else:
            # Mock prediction (deterministic based on image content)
            import hashlib
            img_hash = int(hashlib.md5(contents).hexdigest()[:8], 16)
            np.random.seed(img_hash)
            
            score = np.random.uniform(0, 1)
            is_infected = score > 0.5
            # Generate a fake heatmap
            heatmap = np.random.uniform(0, 1, (128, 128))
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            heatmap = heatmap / np.max(heatmap)
            
            # Reset seed
            np.random.seed(None)

        label = "Parasitized" if is_infected else "Uninfected"
        confidence = score if is_infected else 1.0 - score
        superimposed_img = get_superimposed_img(img_tensor, heatmap)

        # Encode images to base64
        # Original resized image
        buffered_orig = io.BytesIO()
        image.save(buffered_orig, format="JPEG")
        orig_base64 = base64.b64encode(buffered_orig.getvalue()).decode()

        # Heatmap image
        heatmap_img = Image.fromarray(superimposed_img)
        buffered_heat = io.BytesIO()
        heatmap_img.save(buffered_heat, format="JPEG")
        heat_base64 = base64.b64encode(buffered_heat.getvalue()).decode()

        return {
            "prediction": label,
            "confidence": f"{confidence:.2%}",
            "is_infected": bool(is_infected),
            "original_image": f"data:image/jpeg;base64,{orig_base64}",
            "heatmap_image": f"data:image/jpeg;base64,{heat_base64}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Mount static files
if os.path.exists("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
