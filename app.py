from flask import Flask, render_template, request, jsonify
import pickle
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import nltk
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
import threading
from flask import Response


from gtts import gTTS
import os

from camera import VideoCamera
import pyttsx3
import threading
from googletrans import Translator  # <== new
import random
from flask import Flask, render_template, jsonify
import speech_recognition as sr
from deep_translator import GoogleTranslator
import os
import random
import cv2
import numpy as np

app = Flask(__name__)
video_camera = VideoCamera()
translator = Translator()

# Function to load ISL images (from your module code)
def load_letter_image(letter):
    if letter == " ":
        return np.ones((128, 128, 3), dtype=np.uint8) * 255  # White image for spaces
    folder_path = f'F:/ISL/data/{letter.upper()}'
    
    # Ensure the folder path is correct
    if not os.path.exists(folder_path):
        print(f"Folder not found for letter: {letter.upper()} at path: {folder_path}")
        return None
    
    images = os.listdir(folder_path)
    if images:
        random_image = random.choice(images)
        image_path = os.path.join(folder_path, random_image)
        image = cv2.imread(image_path)
        return image
    else:
        print(f"No images found in folder: {folder_path}")
        return None

def display_images(word):
    word_images = []
    for letter in word:
        image = load_letter_image(letter)
        if image is not None:
            colored_image_area = np.ones((128, 128, 3), dtype=np.uint8) * np.array([255, 228, 196], dtype=np.uint8)
            colored_image_area[0:128, :, :] = image
            labeled_image = np.ones((180, 128, 3), dtype=np.uint8) * np.array([152, 251, 152], dtype=np.uint8)
            labeled_image[0:128, :, :] = colored_image_area
            font_scale = 1.2
            font_thickness = 2
            text_color = (0, 0, 0)
            text_size = cv2.getTextSize(letter.upper(), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = (labeled_image.shape[1] - text_size[0]) // 2
            text_y = 155
            cv2.putText(labeled_image, letter.upper(), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
            word_images.append(labeled_image)
    if word_images:
        word_row = np.hstack(word_images)
        return word_row
    else:
        return None

# Speech-to-text function (using the speech_recognition library)
def speech_to_text():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Adjusting for ambient noise
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="mr-IN")
        return text, 'mr'
    except sr.UnknownValueError:
        try:
            text = recognizer.recognize_google(audio, language="hi-IN")
            return text, 'hi'
        except sr.UnknownValueError:
            return "", ''
    except sr.RequestError:
        return "", ''

# Function to translate text into English
def translate_to_english(text, lang_code):
    translator = GoogleTranslator(source=lang_code, target='en')
    translated_text = translator.translate(text)
    return translated_text

# Flask route for handling speech to ISL
@app.route('/module2')
def module2():
    return render_template('module2.html')  # Render the HTML for module 2

# Flask route to start speech-to-text and display ISL images
@app.route('/start_recording', methods=['POST'])
def start_recording():
    # Step 1: Get user input through speech
    user_input, language_code = speech_to_text()
    
    if user_input:
        # Step 2: Translate to English
        english_text = translate_to_english(user_input, language_code)
        
        # Step 3: Convert words to ISL images (generating corresponding ISL images)
        words = english_text.split()
        word_images = []
        for word in words:
            word_row = display_images(word)
            if word_row is not None:
                word_images.append(word_row)
        
        # Convert images to base64 for rendering in HTML
        import base64
        from io import BytesIO
        
        image_data = []
        for img in word_images:
            _, buffer = cv2.imencode('.jpg', img)
            image_bytes = buffer.tobytes()
            encoded_image = base64.b64encode(image_bytes).decode('utf-8')
            image_data.append(encoded_image)

        return jsonify({
            'message': 'Recording successful',
            'transcribed_text': english_text,
            'image_data': image_data
        })
    else:
        return jsonify({'message': 'Could not recognize speech', 'transcribed_text': ''})


@app.route('/features')
def features():
    return render_template('features.html')  # Ensure 'features.html' matches your template filename

@app.route('/about')
def about():
    return render_template('about.html')  # Ensure 'about.html' is the correct template for the About Us page

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Ensure 'contact.html' is the correct template for the Contact page

@app.route('/module1')
def module1():
    return render_template('module1.html')

video_camera = VideoCamera()
translator = Translator()

sentence = ""

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen(video_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    label = video_camera.get_prediction()

    # Improved suggestions based on label
    suggestions = generate_suggestions(label)
    return jsonify(suggestions)






def generate_suggestions(label):
    # Common words and phrases based on hand gestures (alphabet/labels)
    common_words = {
        'A': ["Apple", "Airplane", "Ambulance", "Hello", "Thank you"],
        'B': ["Ball", "Bat", "Bus", "Sorry", "Please"],
        'C': ["Cat", "Cup", "Cow", "Good morning", "Good night"],
        'D': ["Dog", "Doctor", "Door", "How are you?", "Goodbye"],
        'E': ["Elephant", "Egg", "Engine", "Excuse me", "Welcome"],
        'F': ["Fish", "Fan", "Forest", "I love you", "Take care"],
        'G': ["Goat", "Girl", "Garden", "See you", "Good job"],
        'H': ["House", "Hospital", "Horse", "Have a nice day", "Thank you"],
        'I': ["Ice", "Iron", "Ink", "Sorry", "You're welcome"],
        'J': ["Jug", "Jam", "Jeep", "Please help", "Good afternoon"],
        'K': ["Kite", "King", "Key", "Take care", "Good evening"],
        'L': ["Lion", "Leaf", "Lamp", "How are you?", "Nice to meet you"],
        'M': ["Monkey", "Mango", "Market", "See you later", "Good luck"],
        'N': ["Nest", "Net", "Nurse", "I miss you", "Thank you so much"],
        'O': ["Orange", "Owl", "Office", "Have a good day", "I'm sorry"],
        'P': ["Parrot", "Pen", "Pencil", "Good morning", "How's it going?"],
        'Q': ["Queen", "Quilt", "Quick", "Thank you for helping", "Good work"],
        'R': ["Rabbit", "Rose", "Ring", "What's up?", "Happy to see you"],
        'S': ["Sun", "Star", "School", "Nice to meet you", "Take it easy"],
        'T': ["Tiger", "Table", "Train", "I appreciate it", "I'm happy"],
        'U': ["Umbrella", "Unicorn", "Uniform", "Goodbye", "All the best"],
        'V': ["Van", "Vase", "Vegetables", "Well done", "I believe in you"],
        'W': ["Watch", "Whale", "Window", "You're the best", "Congratulations"],
        'X': ["Xylophone", "X-ray", "Xmas Tree", "Well played", "Good job"],
        'Y': ["Yak", "Yogurt", "Yard", "I'm proud of you", "You're amazing"],
        'Z': ["Zebra", "Zoo", "Zip", "Take care", "Great work"],
    }

    label = label.upper()  # Convert to uppercase to match the dictionary keys
    return common_words.get(label, ["No suggestions available"])  # Default suggestion if no match

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get('text')

    translated = translator.translate(text, src='en', dest='hi')
    return jsonify({"translated_text": translated.text})

from gtts import gTTS
import os

@app.route('/speak', methods=['POST'])
def speak():
    data = request.json
    text = data.get('text')

    def speak_thread():
        tts = gTTS(text=text, lang='hi')
        tts.save("speak.mp3")
        os.system("start speak.mp3")  # Windows will open the file

    threading.Thread(target=speak_thread).start()
    return jsonify({"status": "speaking"})

@app.route('/')
def index():
    return render_template('index.html')  # Serve the index page

@app.route('/start')  # This is where the "Learn More" button will redirect
def home():
    return render_template('home.html')  # Serve the home page after clicking "Learn More"



# Define the route for the options page
@app.route('/options')
def options():
    return render_template('options.html')

if __name__ == '__main__':
    nltk.download('punkt')
    app.run(debug=True)
