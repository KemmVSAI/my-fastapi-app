from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI server is running on Render!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.transpose(img_array, (0, 3, 1, 2))

    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    if img_array.shape != tuple(input_shape):
        raise ValueError(f"Input shape mismatch: expected {input_shape}, but got {img_array.shape}")

    interpreter.set_tensor(input_details[0]['index'], img_array.astype(np.float32))
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions = output_data[0]

    labels = ["Benign keratosis-like lesions", "Basal cell carcinoma", "Actinic keratoses",
              "Vascular lesions", "Melanocytic Nevi", "Melanoma", "Dermatofibroma"]

    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]

    result = {"label": labels[predicted_class], "confidence": float(confidence)}
    return result
