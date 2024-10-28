import os
import time
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import traceback
import cv2
import eventlet
import eventlet.wsgi
from twilio.rest import Client
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit, disconnect
from dotenv import load_dotenv
from ultralytics import YOLO
import google.generativeai as genai
import pathlib
import firebase_admin
from firebase_admin import credentials, storage
import asyncio
from google.generativeai.types import generation_types
from google.oauth2 import service_account
import threading

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SOCKET_SECRET_KEY")

# Use Eventlet for async mode
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    ping_interval=25,
    ping_timeout=60,
    async_mode="eventlet",
    max_http_buffer_size=100000000,
    logger=True,
    engineio_logger=True,
    path="/socket.io",
)
def load_models():
    global injury_model, wound_model
    injury_model = YOLO("./models/injury_detection.pt")
    wound_model = YOLO("./models/wound_detection.pt")
    print("Models loaded successfully.")

# Start loading models in a background thread
threading.Thread(target=load_models).start()

# Dictionary to hold recent detections
recent_detections = {}
detection_sent = False


@app.route("/")
def index():
    return render_template("test.html")


@socketio.on("connect")
def handle_connect():
    global detection_sent
    detection_sent = False
    print("Client connected")
    emit("hello", {"message": "Hello, World!"})


@socketio.on("disconnect")
def handle_disconnect():
    global detection_sent
    detection_sent = False
    print("Client disconnected")


@socketio.on("injury_detection_request")
def handle_injury_detection(data):
    global recent_detections, detection_sent

    if detection_sent:
        print("Injury already detected, no further responses will be sent.")
        return

    print("Injury detection request received")

    try:
        if not data:
            emit(
                "injury_detection_response",
                {"error": "No image data provided"},
                to=request.sid,
            )
            return

        # Read the image and ensure it's in RGB
        image = Image.open(BytesIO(data)).convert("RGB")
        frame = np.array(image)

        # Convert frame to BGR for OpenCV processing
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run injury detection using the BGR image
        injury_detections = injury_model.predict(frame_bgr, conf=0.5, save=False)

        # Ensure there is at least one detection
        if injury_detections and len(injury_detections[0].boxes) > 0:
            print("Injury detected")
            box = injury_detections[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop the detected injury area from the original RGB frame
            detected_injury_frame = frame[y1:y2, x1:x2]

            # Save the cropped injury image for debugging
            injury_debug_image = Image.fromarray(detected_injury_frame)
           
             # Convert the entire frame (no cropping) to base64
            buffered = BytesIO()
            Image.fromarray(frame).save(buffered, format="JPEG")  # Using original frame
            injury_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # Generate a unique request ID
            request_id = str(int(time.time() * 1000))
            recent_detections[request_id] = injury_image_base64

            # Emit detection result to client
            emit(
                "injury_detection_response",
                {
                    "message": "Injury detected",
                    "detection": True,
                    "request_id": request_id,
                    "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                },
            )
            detection_sent = True
            disconnect()
        else:
            # Emit no injury detected message
            emit(
                "injury_detection_response",
                {"message": "No injury detected", "detection": False},
            )

    except Exception as e:
        print("Exception occurred:")
        traceback.print_exc()
        emit("injury_detection_response", {"error": str(e)})


@app.route("/wound-detection-request", methods=["POST"])
async def wound_detection_function():
    print("Wound detection request received")
    final_val = ""
    global recent_detections
    data = request.get_json()

    # Check for request_id, username, and contact in data
    if "request_id" not in data or "username" not in data or "contact" not in data:
        return jsonify({"error": "Missing request_id, username, or contact"}), 400

    # Retrieve the values from the JSON payload
    request_id = data["request_id"]
    username = data["username"]
    contact = data["contact"]

    # Retrieve the stored injury image from recent_detections
    if request_id not in recent_detections:
        return jsonify({"error": "No image found for provided request_id"}), 400

    # Decode the base64 image
    injury_image_base64 = recent_detections.pop(request_id)  # Remove after retrieving
    injury_image_data = base64.b64decode(injury_image_base64)
    injury_image = Image.open(BytesIO(injury_image_data)).convert("RGB")
    injury_frame = np.array(injury_image)
    # injury_frame is in RGB format

    try:
        # Convert injury_frame to BGR for OpenCV
        injury_frame_bgr = cv2.cvtColor(injury_frame, cv2.COLOR_RGB2BGR)

        # Run wound classification
        wound_classification = wound_model.predict(
            injury_frame_bgr, conf=0.5, save=False
        )

        if wound_classification and len(wound_classification[0].boxes) > 0:
            # Get confidence scores for each detected box
            confidences = wound_classification[0].boxes.conf.cpu().numpy()
            wound_type = wound_classification[0].names[np.argmax(confidences)]

            final_val = upload_to_gemini(injury_frame, wound_type)
            if final_val == "No response available":
                if wound_type == 'cut':
                    final_val = "Apply pressure to stop bleeding, clean the wound, and cover with a sterile bandage."
                elif wound_type == 'abrasions':
                    final_val = "Clean the wound gently with water, apply an antibiotic ointment, and cover with a bandage."
                elif wound_type == 'burns':
                    final_val = "Cool the burn with cool (not cold) water, cover with a sterile bandage, and avoid popping blisters."
                elif wound_type == 'laseration':
                    final_val = "Clean the wound, apply pressure if bleeding, and consider seeking medical attention if deep."
                elif wound_type == 'stab_wound':
                    final_val = "Apply direct pressure to stop bleeding, cover with a sterile bandage, and seek immediate medical attention."
                elif wound_type == 'ingrown_nail':
                    final_val = "Soak the foot in warm water, gently lift the nail, and place a small piece of cotton under it to reduce pressure."
                elif wound_type == 'bruises':
                    final_val = "Apply ice to the bruised area for 15-20 minutes to reduce swelling. Keep the area elevated if possible."
                else:
                    final_val = "No specific advice available. Ensure the wound is clean and consider seeking medical attention."

            
            image_url = upload_to_firebase(injury_frame)
            await  send_whatsapp_message(image_url, final_val, wound_type, contact, username)
            recent_detections.clear()
        else:
            wound_type = "Unknown"
            recent_detections.clear()

        return jsonify({"message": f"Detected wound type is {wound_type}  and  {final_val}"}), 200

    except Exception as e:
        print("Exception occurred:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


async def send_whatsapp_message(image_url, gemini_response, wound_type, phone_number, name):
    # Twilio credentials
    account_sid = os.getenv("TWILLIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILLIO_AUTH_TOKEN")
    twilio_whatsapp_number = os.getenv("TWILLIO_WHATSAPP_NUMBER")

    # Initialize the Twilio client
    client = Client(account_sid, auth_token)

    # Format the message (combining text and image)
    message_body = (
        f"Dear {name},\n\n"
        f"I wanted to inform you that has an injury identified as {wound_type}. "
        f"The wound appears to be {gemini_response}.\n\n"
        "Please monitor the situation and consider seeking medical attention if it worsens.\n\n"
        f"Wound Type: {wound_type}"
    )

    print("Sending WhatsApp message with text and image...")
    try:
        # Send both text and image in the same message
        message = client.messages.create(
            body=message_body,
            media_url=[image_url],
            from_=twilio_whatsapp_number,
            to=f"whatsapp:{phone_number}",
        )
        print(f"Message with text and image sent successfully: {message.sid}")

    except Exception as e:
        print(f"Failed to send WhatsApp message: {e}")


def upload_to_gemini(injury_frame, wound):
    genai.configure(api_key=os.getenv("GEN_API"))

    # Configuration settings
    generation_config = {
        "temperature": 2,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        safety_settings=safety_settings,
    )

    # Convert injury_frame to image bytes
    pil_image = Image.fromarray(injury_frame)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # Encode image data as base64
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data = {"mime_type": "image/jpeg", "data": image_base64}

    # Start conversation with image data
    convo = model.start_chat(history=[])
    convo.send_message(
        f"A wound has been detected and classified as {wound}. "
        f"Please provide a simple, practical suggestion for treating this wound in fewer than 50 words. "
        f"For example, for bruises or abrasions, recommend cleaning and applying ice. "
        f"For burns, suggest cooling with water and applying a bandage. "
        f"For deep cuts or stab wounds, suggest applying pressure and seeking medical attention. "
        f"Do not include any disclaimers or advice such as 'consult a professional.' Just give practical advice."
    )
    # Attempt to send the image
    try:
        convo.send_message(image_data)
    except generation_types.BlockedPromptException as e:
        print("Image was blocked by Gemini API.")
        # Log or handle the blocked prompt gracefully
    
    return convo.last.text if convo.last else "No response available"



def upload_to_firebase(injury_frame):
    firebase_key_path = "./credentials.json"
    
    # Initialize Firebase only if not already initialized
    if not firebase_admin._apps:
        cred = credentials.Certificate(firebase_key_path)
        firebase_admin.initialize_app(cred, {"storageBucket": "blindsafe-b24ff.appspot.com"})
    
    bucket = storage.bucket()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"wound_images/{timestamp}_injury.jpg"

    # Convert injury_frame to image bytes
    pil_image = Image.fromarray(injury_frame)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    image_bytes = buffered.getvalue()

    # Upload image to Firebase
    blob = bucket.blob(filename)
    blob.upload_from_string(image_bytes, content_type="image/jpeg")
    blob.make_public()

    print(blob.public_url)
    return blob.public_url

if __name__ == "__main__":
    print("Starting Socket.IO server")
    socketio.run(app, host="0.0.0.0", port=5000)
