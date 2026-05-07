import numpy as np
from PIL import Image
import io
from predictor import MalariaPredictor

def test_predictor():
    predictor = MalariaPredictor()
    
    # 1. Test with a solid red image
    red_img = Image.new('RGB', (128, 128), color='red')
    img_byte_arr = io.BytesIO()
    red_img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    print("Testing with solid red image...")
    result = predictor.predict(img_bytes)
    print(f"Result: {result}")
    
    if "error" in result:
        print("PASS: Solid red image rejected as expected.")
    else:
        print("FAIL: Solid red image was not rejected.")

    # 2. Test with a random noise image (should have texture but maybe not right color/shape)
    noise_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    noise_pil = Image.fromarray(noise_img)
    img_byte_arr = io.BytesIO()
    noise_pil.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    print("\nTesting with random noise image...")
    result = predictor.predict(img_bytes)
    print(f"Result: {result.get('error', 'Success (as predicted)')}")

if __name__ == "__main__":
    test_predictor()
