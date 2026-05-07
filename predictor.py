import os
import io
import cv2
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
import hashlib

class MalariaPredictor:
    def __init__(self, model_path="malaria_model_validated.h5"):
        self.model_path = model_path
        self.model = None
        self.using_mock = False
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
                self.using_mock = False
            except Exception as e:
                print(f"Error loading model: {e}")
                self.using_mock = True
        else:
            print(f"Model file {self.model_path} not found. Using mock mode.")
            self.using_mock = True

    def is_valid_cell_image(self, image_pil):
        """
        Heuristic check to see if the image is likely a malaria cell slide.
        Checks for color distribution, variance, and texture.
        """
        # Convert to CV2 format (BGR)
        open_cv_image = np.array(image_pil)
        if len(open_cv_image.shape) == 2: # Grayscale
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_GRAY2BGR)
        else:
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

        # 1. Color Variance Check: Reject solid colors (like a pure red image)
        # Calculate standard deviation across all pixels
        std_dev = np.std(open_cv_image)
        if std_dev < 10: # Very low variance means it's likely a solid color
            return False, "Image is too uniform (solid color detected)."

        # 2. Edge/Texture Check: Real microscopic images have edges
        # Use Laplacian variance to detect texture
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 20: # Low variance in Laplacian means it's very blurry or blank
            return False, "Image lacks microscopic detail or texture."

        # 3. Stain Color Check: Malaria slides are usually stained (Purple/Pink/Blue)
        # Typical stains (Giemsa/H&E) have lower Green values compared to Red and Blue.
        b, g, r = cv2.split(open_cv_image)
        mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
        
        # Check for overly dominant pure colors (Red, Green, Blue)
        if mean_r > 220 and mean_g < 60 and mean_b < 60:
            return False, "Invalid image: Dominant red color detected (not a cell slide)."
        if mean_g > 200 and mean_r < 100 and mean_b < 100:
            return False, "Invalid image: Dominant green color detected (not a cell slide)."
        
        # Most malaria slides have a purple/pinkish hue. 
        # If the image is too far from this profile (e.g., very yellowish or greenish), flag it.
        # Simple heuristic: In stained cells, Red and Blue are usually higher than Green.
        if mean_g > mean_r and mean_g > mean_b:
            return False, "Invalid image: Color profile does not match a stained microscopic slide."

        return True, "Valid"

    def make_gradcam_heatmap(self, img_array, last_conv_layer_name, pred_index=None):
        grad_model = tf.keras.models.Model(
            [self.model.inputs], [self.model.get_layer(last_conv_layer_name).output, self.model.output]
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

    def get_superimposed_img(self, img_tensor, heatmap):
        heatmap = np.uint8(255 * heatmap)
        jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
        
        original_img = np.uint8(255 * img_tensor[0])
        superimposed_img = jet * 0.4 + original_img * 0.6
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        return superimposed_img

    def predict(self, image_bytes):
        # Open and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Validate image
        is_valid, reason = self.is_valid_cell_image(image)
        if not is_valid:
            return {"error": reason}

        image_resized = image.resize((128, 128))
        img_array = np.array(image_resized).astype('float32') / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        if not self.using_mock and self.model is not None:
            # Real prediction
            preds = self.model.predict(img_tensor)
            score = float(preds[0][0])
            is_infected = score > 0.5
            
            # Find last conv layer for Grad-CAM
            try:
                last_conv = [l.name for l in reversed(self.model.layers) if 'conv2d' in l.name][0]
                heatmap = self.make_gradcam_heatmap(img_tensor, last_conv)
            except:
                # Fallback if no conv layer found
                heatmap = np.zeros((128, 128))
        else:
            # Mock logic
            img_hash = int(hashlib.md5(image_bytes).hexdigest()[:8], 16)
            np.random.seed(img_hash)
            score = np.random.uniform(0, 1)
            is_infected = score > 0.5
            heatmap = np.random.uniform(0, 1, (128, 128))
            heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
            heatmap = heatmap / (np.max(heatmap) + 1e-10)
            np.random.seed(None)

        label = "Parasitized" if is_infected else "Uninfected"
        confidence = score if is_infected else 1.0 - score
        superimposed_img = self.get_superimposed_img(img_tensor, heatmap)

        # Encode images to base64
        buffered_orig = io.BytesIO()
        image_resized.save(buffered_orig, format="JPEG")
        orig_base64 = base64.b64encode(buffered_orig.getvalue()).decode()

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
