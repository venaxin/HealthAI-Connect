import warnings
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
print("Loading libraries...")
from flask import Flask, render_template, request, redirect,jsonify, url_for, session
import pandas as pd
import numpy as np
import joblib 
import json
import pickle as pk
import re
import csv
import scipy
from pathlib import Path
from datetime import datetime
print('Loading ----25% Done')
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import google.generativeai as genai
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
print('Loading --------50% Done')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
print('Loading ------------75% Done')
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
print("     Libraries Loaded Successfully")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.simplefilter("ignore", category=DeprecationWarning)


# Mute all warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = 'Secret@key'  # Change this to a secure key in a production environment
app.config['SECRET_KEY'] = 'secret!'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'  # Use SQLite for simplicity
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
socketio = SocketIO(app)


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    content = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize the database within the script
with app.app_context():
    db.create_all()

@app.route('/chat')
def chat():
    username = request.args.get('username')
    messages = Message.query.all()
    return render_template('chat.html', messages=messages, username=username)

@socketio.on('message')
def handle_message(data):
    username = data.get('username')
    content = data['content']
    
    new_message = Message(username=username, content=content)
    db.session.add(new_message)
    db.session.commit()
    
    emit('message', {'username': username, 'content': content}, broadcast=True)

@app.route('/clear_chats', methods=['POST'])
def clear_chats():
    if request.method == 'POST':
        Message.query.delete()  # Delete all messages
        db.session.commit()
        return redirect(url_for('chat', username=request.args.get('username')))
    else:
        # Return a 405 Method Not Allowed response for other methods
        return 'Method Not Allowed', 405

# CSV file paths
patient_appointment_csv_file_path = 'database/Accepted_appointments.csv'
patient_csv_file_path = 'database/credentials.csv'
doctor_csv_file_path = 'database/doctor_credentials.csv'
admin_csv_file_path = 'database/admin_credentials.csv'
appointment_csv_file_path = 'database/appointments.csv'

# Create CSV files if they don't exist
Path(patient_csv_file_path).touch()
Path(doctor_csv_file_path).touch()
Path(admin_csv_file_path).touch()
Path(appointment_csv_file_path).touch()
Path(patient_appointment_csv_file_path).touch()
# Load existing users from CSV files
with open(patient_csv_file_path, 'r') as file:
    patient_reader = csv.reader(file)
    patients = {row[0]: row[1] for row in patient_reader}

with open(doctor_csv_file_path, 'r') as file:
    doctor_reader = csv.reader(file)
    doctors = {row[0]: row[1] for row in doctor_reader}

with open(admin_csv_file_path, 'r') as file:
    admin_reader = csv.reader(file)
    admins = {row[0]: row[1] for row in admin_reader}

print('Loading embedding models...')
# Chatbot
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="deepset/sentence_bert")
knowledge_base = FAISS.load_local('embed',embeddings)
llm = OpenAI(model="gpt-3.5-turbo-instruct",temperature=0.9, max_tokens=500)
#llm=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
print("     Loaded embedding models")
print('Loading  ML and DL models...')
# Brain Tumor Detection
tumor_model =load_model('models/BrainTumor15Epochscategorical.h5')

# Pneumonia Detection
pneumonia_model = load_model('models/chest_xray.h5')

# Review based disease prediction
MODEL_PATH = 'models/passmodel.pkl'
TOKENIZER_PATH ='models/tfidfvectorizer.pkl'
DATA_PATH ='data/drugsComTrain.csv'
vectorizer = joblib.load(TOKENIZER_PATH)
model = joblib.load(MODEL_PATH)
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
rawtext = ""

# Prediction Using disease symptoms
df1 = pd.read_csv('data/Symptom-severity.csv')
df1['Symptom'] = df1['Symptom'].str.replace('_',' ')
discrp = pd.read_csv("data/symptom_Description.csv")
ektra7at = pd.read_csv("data/symptom_precaution.csv")
df = pd.read_csv('data/dataset.csv')
with open('models/symptoms.pkl', 'rb') as file:
    diseaseModel = pk.load(file)
with open('models/symptoms.json', 'r') as json_file:
    symptoms = json.load(json_file)
print('     Loaded all models')
print('Starting app...')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/symptompredict', methods=["GET",'POST'])
def symptompredict():
    if request.method=="POST":  
        selected_symptoms = [
            request.form.get('symptom1'),
            request.form.get('symptom2'),
            request.form.get('symptom3'),
            request.form.get('symptom4'),
            request.form.get('symptom5'),
            request.form.get('symptom6'),
        ]
        psymptoms = selected_symptoms
        selected_symptom=[]
        # Perform the prediction using the loaded model
        a = np.array(df1["Symptom"])
        b = np.array(df1["weight"])
        for j in range(len(psymptoms)):
            for k in range(len(a)):
                if psymptoms[j]==a[k]:
                    selected_symptom.append(a[k])
                    psymptoms[j]=b[k]
        psy = [psymptoms]
        pred2 = diseaseModel.predict(psy)
        decision_function_scores = diseaseModel.predict_proba(psy)
    
    # Get the index of the predicted class
        predicted_class_index = np.argmax(decision_function_scores)
        confidence_percentage = round(decision_function_scores[0, predicted_class_index] * 100, 2)

        # Additional code for getting disease description and recommendations
        disp = discrp[discrp['Disease'] == pred2[0]]
        disp = disp.values[0][1]
        recomnd = ektra7at[ektra7at['Disease'] == pred2[0]]
        c = np.where(ektra7at['Disease'] == pred2[0])[0][0]
        precaution_list = []
        for i in range(1, len(ektra7at.iloc[c])):
            precaution_list.append(ektra7at.iloc[c, i])

        return render_template('symptom.html', disease=pred2[0], description=disp, precautions=precaution_list ,symptoms=symptoms,confidence=confidence_percentage , selected_symptoms=selected_symptom)
    return render_template('symptom.html',symptoms=symptoms)


#chatbot
@app.route('/send', methods=['POST'])
def send():
    user_question = request.json.get('message')
    response = None
    #user_question="What is the best way to get a job in the tech industry?"
    print(user_question)
    if user_question:
        try: 
            # Attempt to perform similarity search
            google_gemini=genai.GenerativeModel('gemini-pro')
            docs = knowledge_base.similarity_search(user_question)
            print(docs[0])
            doc=f" {docs[0]}"
            # Assuming 'llm' is a pre-trained language model If nothing related to the question in the prompt then you can't answer the question .
            #chain = load_qa_chain(llm=llm, chain_type="stuff")
            #with get_openai_callback() as cb:
                # Attempt to run the QA chain
             #   response = chain.run(input_documents=docs, question=user_question,return_only_outputs=True)
              #  print(cb)
            PROMPT="""You are an expert in medical and healthcare knowledge and your name is medibot. If you are asked "who are you" you can say your name and profession. and you can reply to thank you also.
            you need to answer the following question from the given information only and provide the best possible answer from given info only.
            if there is any related word or info in the prompt then you can answer the question from that or you can answer by your own.   
            Your answer should be very precise and in one para and to the point dont give any irrelevant information. your answer should be like you are telling not giving from any resources"""
            question=PROMPT+user_question+doc 
            responses= google_gemini.generate_content(question)
            response=responses.text
            return jsonify({'response': response})

        except ValueError as e:
            # Handle missing document gracefully
            print(f"Error: {e}")
            return jsonify({'response': 'Error processing question'})

        except Exception as e:
            # Log any other unexpected exception
            print(f"Unexpected error: {e}")
            return jsonify({'response': 'An unexpected error occurred'})

    else:
        return jsonify({'response': 'No question provided'})

# review based disease prediction

@app.route('/reviewpredict', methods=["GET", "POST"])
def reviewpredict():
    
    if request.method == 'POST':
        raw_text = request.form['rawtext']

        if raw_text != "":
            clean_text = cleanText(raw_text)
            clean_lst = [clean_text]
            text_vectorized = vectorizer.transform(clean_lst)
            # Get the decision values (confidence scores) from the model
            decision_values = model.decision_function(text_vectorized)

            # Apply the sigmoid function to squash decision values into [0, 1] range
            sigmoid_scores = scipy.special.expit(decision_values)

            # Display the predicted class and its confidence scores
            predicted_class = model.predict(text_vectorized)[0]
            confidence = round(sigmoid_scores[0, model.classes_ == predicted_class][0] * 100, 2) 
            df = pd.read_csv(DATA_PATH)
            top_drugs = top_drugs_extractor(predicted_class, df)
            return render_template('review.html', rawtext=raw_text, result=predicted_class, top_drugs=top_drugs,confidence=confidence)
        else:
            raw_text = "There is no text to select"

    return render_template('review.html', rawtext=rawtext)


   


def cleanText(raw_review):
    # 1. Delete HTML 
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords 
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmitization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return( ' '.join(lemmitize_words))


def top_drugs_extractor(condition,df):
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst



# Nutritionist Doctor

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([image[0], prompt+input])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file:
        bytes_data = uploaded_file.read()
        image_parts = [{"mime_type": uploaded_file.mimetype, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")


@app.route('/nutri', methods=['GET','POST'])
def nutri():
    if request.method=='POST':
        input_prompt = """
        You are an expert in nutritionist where you need to see the food items from the image
        and calculate the total calories in the food items.
        """
        uploaded_file = request.files['file']
        input_text = request.form['description']
        
        image_data = input_image_setup(uploaded_file)
        response = get_gemini_response(input_text, image_data, input_prompt)
        return  response
    return render_template('nutri.html')


@app.route('/healthcare', methods=['GET'])
def healthcare():
    return render_template('healthimaging.html')
@app.route('/healthimaging_patient', methods=['GET'])
def healthimaging_patient():
    return render_template('healthimaging_patient.html')
#Brain Tumor Detection
def get_className(classNo):
    class_index = np.argmax(classNo)
    if class_index==0:
	    return "No Brain Tumor Detected."
    elif class_index==1:
    	return """Brain tumor detected. We strongly advise you to consult with a doctor immediately for further evaluation and treatment options."""


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=tumor_model.predict(input_img)
    return result


@app.route('/tumor', methods=['GET'])
def tumor():
    return render_template('tumor1.html')


@app.route('/predict_tumor', methods=['GET', 'POST'])
def predict_tumor():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads/tumor', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None

# pneumonia

def predict_pneumonia(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)
    classes = pneumonia_model.predict(img_data)
    result = int(classes[0][0])
    return result

@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/predict_neumonia', methods=['POST'])
def predict_neumonia():
    if 'file' not in request.files:
        print('No file part')
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        print('No selected file')
        return jsonify({'error': 'No selected file'})
    if file:
        file_path = os.path.join('uploads/pneumonia', file.filename)
        file.save(file_path)
        result = predict_pneumonia(file_path)
        if result == 0:
            prediction = "Person is affected by PNEUMONIA."
        else:
            prediction = "Your Lungs are not affected by PNEUMONIA."
        print(prediction)   
        return prediction
    




@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in patients and patients[username] == password:
            session['username'] = username
            session['user_type'] = 'patient'
            return redirect(url_for('dashboard'))
        elif username in doctors and doctors[username] == password:
            session['username'] = username
            session['user_type'] = 'doctor'
            return redirect(url_for('doctor_dashboard'))
        elif username in admins and admins[username] == password:
            session['username'] = username
            session['user_type'] = 'admin'
            return redirect(url_for('admin_dashboard'))
        else:
            return render_template('login.html', message='Invalid credentials. Try again.')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        user_type = session.get('user_type')
        if user_type == 'patient':
            patient_username = session['username']
            appointments = get_patient_appointments(patient_username)
            return render_template('patient_dashboard.html', username=patient_username, appointments=appointments,doctors=doctors)
        elif user_type == 'doctor':
            # Load doctor-specific data and pass it to the template
            return render_template('doctor_dashboard.html', username=session['username'], appointments=get_doctor_appointments(session['username']))
    else:
        return redirect(url_for('home'))

def get_patient_appointments(patient_username):
    try:
        with open(patient_appointment_csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            appointments = [row for row in reader if row['patient'] == patient_username]
        return appointments
    except FileNotFoundError:
        return []

    
@app.route('/admin_dashboard')
def admin_dashboard():
    if 'username' in session and session['user_type'] == 'admin':
        with open(patient_csv_file_path, 'r') as file:
            patient_reader = csv.DictReader(file)
            patientss = [row for row in patient_reader]
        with open(doctor_csv_file_path, 'r') as file:
            doctor_reader = csv.DictReader(file)
            doctorss = [row for row in doctor_reader]
        return render_template('admin_dashboard.html', username=session['username'], patientss=patientss, doctorss=doctorss)
    else:
        return redirect(url_for('home'))
    
@app.route('/admin_delete_user', methods=['POST'])
def admin_delete_user():
    patients=''
    doctors=''
    if 'username' in session and session['user_type'] == 'admin':
        user_type = request.form.get('user_type')
        username_to_delete = request.form.get('username')
        print(user_type, username_to_delete)
        if user_type and username_to_delete:
            if user_type == 'patient':
                with open(patient_csv_file_path, 'r') as file:
                    patient_reader = csv.DictReader(file)
                    patientss = [row for row in patient_reader]
                # Use a list comprehension to exclude the user to delete
                patient = [row for row in patientss if row['username'] != username_to_delete]
                print(patients)
                print('deleting patient')
                save_userss_to_csv(patient, patient_csv_file_path)
            elif user_type == 'doctor':
                with open(doctor_csv_file_path, 'r') as file:
                    doctor_reader = csv.DictReader(file)
                    doctorss = [row for row in doctor_reader]
                doctor = [row for row in doctorss if row['username'] != username_to_delete]
                print('deleting doctor')
                save_userss_to_csv(doctor, doctor_csv_file_path)

    return redirect(url_for('admin_dashboard'))

def save_userss_to_csv(users, file_path):
    with open(file_path, 'w', newline='') as file:
        fieldnames = ['username', 'password']  # Replace with actual field names
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(users)
@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('user_type', None)
    return redirect(url_for('home'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user_type = request.form.get('user_type')

        if username and password and user_type:
            if user_exists(username):
                return render_template('register.html', message='Username already exists. Try another one.')

            if user_type == 'patient':
                patients[username] = password
                save_users_to_csv(patients, patient_csv_file_path)
            elif user_type == 'doctor':
                doctors[username] = password
                save_users_to_csv(doctors, doctor_csv_file_path)

            return redirect(url_for('home'))
        else:
            return render_template('register.html', message='Username, password, and user type are required.')

    return render_template('register.html')

def user_exists(username):
    # Check if the username exists in patients or doctors
    return username in patients or username in doctors

@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    if request.method == 'POST':
        patient_username = session.get('username')
        doctor_username = request.form.get('doctor')
        date = request.form.get('date')
        time = request.form.get('time')

        if patient_username and doctor_username and date and time:
            appointment_data = f"{doctor_username},{patient_username},{date},{time}"
            save_appointment_to_csv(appointment_data, appointment_csv_file_path)
            return redirect(url_for('dashboard'))
        else:
            return render_template('patient_dashboard.html', username=session['username'], message='All fields are required.')

def save_users_to_csv(users, file_path):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        for username, password in users.items():
            writer.writerow([username, password])


def save_appointment_to_csv(appointment_data, file_path):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(appointment_data.split(','))


def get_doctor_appointments(doctor_username):
    try:
        with open(appointment_csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            appointments = [row for row in reader if row['doctor'] == doctor_username]
        return appointments
    except FileNotFoundError:
        return []


@app.route('/doctor_dashboard')
def doctor_dashboard():
    if 'username' in session and session['user_type'] == 'doctor':
        doctor_username = session['username']
        appointments = get_doctor_appointments(doctor_username)
        accepted_appointments = get_accepted_appointments(doctor_username)
        return render_template('doctor_dashboard.html', username=doctor_username, appointments=appointments,
                               accepted_appointments=accepted_appointments)
    else:
        return redirect(url_for('home'))


def get_accepted_appointments(doctor_username):
    try:
        with open(patient_appointment_csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            appointments = [dict(row) for row in reader if row['doctor'] == doctor_username and row['condition'] == 'accept']
        return appointments
    except FileNotFoundError:
        return []


@app.route('/complete_appointment/<int:appointment_id>', methods=['POST'])
def complete_appointment(appointment_id):
    if 'username' in session and session['user_type'] == 'doctor':
        doctor_username = session['username']
        accepted_appointments = get_accepted_appointments(doctor_username)

        if 0 <= appointment_id < len(accepted_appointments):
            appointment = accepted_appointments[appointment_id]

            # Remove the appointment from the CSV file
            remove_appointment_from_csv(doctor_username, appointment)

    return redirect(url_for('doctor_dashboard'))


def remove_appointment_from_csv(doctor_username, appointment):
    try:
        with open(patient_appointment_csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            appointments = [row for row in reader if
                            row['doctor'] == doctor_username and row['condition'] == 'accept' and (
                                    row['patient'], row['date'], row['time']) != (
                                    appointment['patient'], appointment['date'], appointment['time'])]

        with open(patient_appointment_csv_file_path, 'w', newline='') as file:
            fieldnames = ['doctor', 'patient', 'date', 'time', 'condition']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(appointments)
    except FileNotFoundError:
        pass


def remove_appointments_from_csv(doctor_username, appointment):
    try:
        with open(appointment_csv_file_path, 'r') as file:
            reader = csv.DictReader(file)
            appointments = [
                row for row in reader if row['doctor'] == doctor_username and
                                         (row['doctor'] != appointment['doctor'] or
                                          row['patient'] != appointment['patient'] or
                                          row['date'] != appointment['date'] or
                                          row['time'] != appointment['time'])
            ]

        with open(appointment_csv_file_path, 'w', newline='') as file:
            fieldnames = ['doctor', 'patient', 'date', 'time']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(appointments)
    except FileNotFoundError:
        pass


@app.route('/accept_appointment/<int:appointment_id>', methods=['POST'])
def accept_appointment(appointment_id):
    print("Entering accept_appointment route")

    if 'username' in session and session['user_type'] == 'doctor':
        print("User is a doctor and logged in")
        doctor_username = session['username']
        appointments = get_doctor_appointments(doctor_username)
        action = request.form['action']

        if request.method == 'POST':
            print("Handling POST request")
            # Process the form submission only for POST requests
            if 0 <= appointment_id < len(appointments):
                print(f"Valid appointment_id: {appointment_id}")
                appointment = appointments[appointment_id]
                print(len(appointment))
                if len(appointment) >= 4:
                    if action == 'accept':
                        save_appointments_to_csv(appointments, patient_appointment_csv_file_path)
                        print("Appointment accepted")
                        # Get additional appointment details
                        doctor_name = appointment['doctor']
                        date = appointment['date']
                        time = appointment['time']
                        patient_name = appointment['patient']
                        print(f"Appointment details: {date}, {time}, {patient_name}")
                        # Save the accepted appointment details to patient_appointments.csv
                        save_accepted_appointment(doctor_name, date, time, patient_name, 'accept',
                                                 patient_appointment_csv_file_path)
                        remove_appointments_from_csv(doctor_name, appointment)
                    elif action == 'reject':
                        remove_appointments_from_csv(doctor_username, appointment)
                        print("Appointment rejected")
                return redirect(url_for('doctor_dashboard'))
    return redirect('doctor_dashboard')


def save_accepted_appointment(doctor_name, date, time, patient_name, condition, file_path):
    fieldnames = ['doctor', 'patient', 'date', 'time', 'condition']
    # Check if the file exists
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        # Write header only if the file is newly created
        if not file_exists:
            writer.writeheader()
        # Write the accepted appointment details with the condition set to 'accept'
        writer.writerow({'doctor': doctor_name, 'patient': patient_name, 'date': date, 'time': time, 'condition': condition})


def save_appointments_to_csv(appointments, file_path):
    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        for row in appointments:
            writer.writerow(row)

warnings.filterwarnings("default")
if __name__ == '__main__':
    socketio.run(app, debug=True)
