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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
"""Vector stack (FAISS + HuggingFace embeddings) imported lazily to avoid hard crash
if optional packages are missing in the deployed image. This prevents the whole
app from failing to boot (503) when langchain-community or langchain-huggingface
is absent; vector search features will just be disabled."""
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    _VECTOR_STACK_AVAILABLE = True
except Exception as _vector_err:  # broad so any import side-effect failure is caught
    print(f"Vector stack unavailable, continuing without retrieval: {_vector_err}")
    FAISS = None  # type: ignore
    HuggingFaceEmbeddings = None  # type: ignore
    _VECTOR_STACK_AVAILABLE = False
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
try:
    import cv2  # OpenCV is optional; if system libs missing we continue without it
    _CV2_AVAILABLE = True
except Exception as _cv2_err:
    print(f"OpenCV not available, continuing without it: {_cv2_err}")
    cv2 = None  # type: ignore
    _CV2_AVAILABLE = False
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
# Use .env or defaults
from dotenv import load_dotenv as _load
_load()
app.secret_key = os.getenv('SECRET_KEY', 'Secret@key')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'secret!')
# Robust DATABASE_URL resolution: treat unset or blank as missing
_raw_db_url = os.getenv('DATABASE_URL', '').strip()
if not _raw_db_url:
    _raw_db_url = 'sqlite:////tmp/chat.db'
elif _raw_db_url.startswith('sqlite:///') and not _raw_db_url.startswith('sqlite:////'):
    # Normalize absolute /tmp path if someone used 3 slashes
    if _raw_db_url.endswith('/tmp/chat.db'):
        _raw_db_url = 'sqlite:////tmp/chat.db'
app.config['SQLALCHEMY_DATABASE_URI'] = _raw_db_url
print(f"Using database URL: {app.config['SQLALCHEMY_DATABASE_URI']}")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
# Use eventlet async mode in production
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins='*')


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False)
    content = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, index=True)  # patient | doctor | admin

class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor = db.Column(db.String(80), nullable=False, index=True)
    patient = db.Column(db.String(80), nullable=False, index=True)
    date = db.Column(db.String(20), nullable=False)
    time = db.Column(db.String(20), nullable=False)
    age = db.Column(db.String(10))
    phone = db.Column(db.String(30))
    session_purpose = db.Column(db.String(160))
    status = db.Column(db.String(20), nullable=False, default='pending', index=True)  # pending | accept | completed | reject

def _seed_from_csv_if_empty():
    """Populate DB from legacy CSV files if tables empty (first run migration)."""
    # Legacy CSV paths
    legacy_patient = patient_csv_file_path
    legacy_doctor = doctor_csv_file_path
    legacy_admin = admin_csv_file_path
    legacy_appt = appointment_csv_file_path
    legacy_accept = patient_appointment_csv_file_path
    if User.query.count() == 0:
        # Users
        for path, role in [ (legacy_patient,'patient'), (legacy_doctor,'doctor'), (legacy_admin,'admin') ]:
            try:
                if os.path.exists(path) and os.path.getsize(path)>0:
                    dfu = pd.read_csv(path, header=None, names=['username','password'])
                    for _, r in dfu.iterrows():
                        if not User.query.filter_by(username=r.username).first():
                            db.session.add(User(username=r.username, password=r.password, role=role))
            except Exception as e:
                print(f"Seed warn ({role}): {e}")
        db.session.commit()
    if Appointment.query.count() == 0:
        # Pending appointments
        for path, status in [ (legacy_appt,'pending'), (legacy_accept,'accept') ]:
            try:
                if os.path.exists(path) and os.path.getsize(path)>0:
                    cols = ['doctor','patient','date','time','age','phone','session_purpose']
                    # Accepted file may have condition column; handle flexibly
                    dfap = pd.read_csv(path, header=None)
                    # Try to align columns
                    if dfap.shape[1] >= 7:
                        dfap = dfap.iloc[:,0:7]
                        dfap.columns = cols
                        for _, r in dfap.iterrows():
                            db.session.add(Appointment(doctor=r.doctor, patient=r.patient, date=r.date, time=r.time,
                                                       age=r.age, phone=r.phone, session_purpose=r.session_purpose,
                                                       status=status))
                db.session.commit()
            except Exception as e:
                print(f"Seed warn (appointments {status}): {e}")

with app.app_context():
    db.create_all()
    _seed_from_csv_if_empty()

def _user_maps():
    """Return dicts of users by role for legacy code compatibility."""
    users = User.query.all()
    patients_map = {u.username: u.password for u in users if u.role=='patient'}
    doctors_map = {u.username: u.password for u in users if u.role=='doctor'}
    admins_map = {u.username: u.password for u in users if u.role=='admin'}
    return patients_map, doctors_map, admins_map

@app.route('/chat')
def chat():
    username = request.args.get('username')
    p_map, d_map, _a_map = _user_maps()
    messages = Message.query.all()
    return render_template('chat.html', messages=messages, username=username, doctors=d_map, patients=p_map)

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

 
# Legacy CSV (used only for initial seeding now)
patient_csv_file_path = 'database/credentials.csv'  # legacy
doctor_csv_file_path = 'database/doctor_credentials.csv'  # legacy
admin_csv_file_path = 'database/admin_credentials.csv'  # legacy
appointment_csv_file_path = 'database/appointments.csv'  # legacy
patient_appointment_csv_file_path = 'database/Accepted_appointments.csv'  # legacy
DATA_DIR = os.getenv('DATA_DIR', '/tmp/data')
os.makedirs(DATA_DIR, exist_ok=True)
patient_csv_file_path = os.path.join(DATA_DIR, 'credentials.csv')
doctor_csv_file_path = os.path.join(DATA_DIR, 'doctor_credentials.csv')
admin_csv_file_path = os.path.join(DATA_DIR, 'admin_credentials.csv')
appointment_csv_file_path = os.path.join(DATA_DIR, 'appointments.csv')
patient_appointment_csv_file_path = os.path.join(DATA_DIR, 'Accepted_appointments.csv')

# Create CSV files if they don't exist
for file_path in [patient_csv_file_path, doctor_csv_file_path, admin_csv_file_path, appointment_csv_file_path, patient_appointment_csv_file_path]:
    Path(file_path).touch()

patients, doctors, admins = _user_maps()

print('Loading embedding models...')
# Chatbot
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
embeddings = None
knowledge_base = None
if _VECTOR_STACK_AVAILABLE:
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # type: ignore
        knowledge_base = FAISS.load_local('embed', embeddings, allow_dangerous_deserialization=True)  # type: ignore
        print("     Loaded embedding models and FAISS index")
    except Exception as e:
        print(f"Warning: Embeddings/FAISS index not fully loaded: {e}")
        embeddings = None
        knowledge_base = None
else:
    print("     Skipping embeddings/FAISS load (vector stack unavailable)")
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
            docs = []
            if knowledge_base is not None:
                docs = knowledge_base.similarity_search(user_question)
                if docs:
                    print(docs[0])
            doc = f" {docs[0]}" if docs else ""
            # Using Gemini directly without LangChain QA chain
            PROMPT="""You are an expert in medical and healthcare knowledge and your name is medibot. if you are asked question which is not related to the medical field or healthcare field then you can't answer the question."""
            question=PROMPT+user_question 
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

@app.route('/hddp', methods=["GET", "POST"])
def hddp():
    
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
            return render_template('hddp.html', rawtext=raw_text, result=predicted_class, top_drugs=top_drugs,confidence=confidence)
        else:
            raw_text = "There is no text to select"

    return render_template('hddp.html', rawtext=rawtext)

   
@app.route('/spp', methods=["GET",'POST'])
def spp():
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

        return render_template('spp.html', disease=pred2[0], description=disp, precautions=precaution_list ,symptoms=symptoms,confidence=confidence_percentage , selected_symptoms=selected_symptom)
    return render_template('spp.html',symptoms=symptoms)

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
    if class_index == 0:
        return "No Brain Tumor Detected."
    elif class_index == 1:
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

@app.route('/patient_tumor', methods=['GET'])
def patient_tumor():
    return render_template('patient_tumor.html')

@app.route('/predict_tumor', methods=['GET', 'POST'])
def predict_tumor():
    if request.method == 'POST':
        f = request.files['file']

    # Use a writable uploads root (default to /tmp on Cloud Run)
    uploads_root = os.getenv('UPLOAD_DIR', '/tmp/uploads')
    tumor_dir = os.path.join(uploads_root, 'tumor')
    os.makedirs(tumor_dir, exist_ok=True)
    file_path = os.path.join(tumor_dir, secure_filename(f.filename))
    f.save(file_path)
    value = getResult(file_path)
    result = get_className(value)
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
@app.route('/patient_pneumonia')
def patient_pneumonia():
    return render_template('patient_pneumonia.html')
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
        uploads_root = os.getenv('UPLOAD_DIR', '/tmp/uploads')
        pneu_dir = os.path.join(uploads_root, 'pneumonia')
        os.makedirs(pneu_dir, exist_ok=True)
        file_path = os.path.join(pneu_dir, secure_filename(file.filename))
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
            return render_template('patient_dashboard.html', username=patient_username, appointments=appointments, doctors=doctors)
        elif user_type == 'doctor':
            # Load doctor-specific data and pass it to the template
            return render_template('doctor_dashboard.html', username=session['username'], appointments=get_doctor_appointments(session['username']))
    else:
        return redirect(url_for('home'))


def get_patient_appointments(patient_username):
    appts = Appointment.query.filter_by(patient=patient_username).filter(Appointment.status.in_(['pending','accept','completed'])).all()
    return [
        {
            'id': a.id,
            'doctor': a.doctor,
            'patient': a.patient,
            'date': a.date,
            'time': a.time,
            'age': a.age,
            'phone': a.phone,
            'session_purpose': a.session_purpose,
            'status': a.status
        } for a in appts
    ]


@app.route('/admin_dashboard')
def admin_dashboard():
    if 'username' in session and session['user_type'] == 'admin':
        patientss = [{'username': u.username} for u in User.query.filter_by(role='patient').all()]
        doctorss = [{'username': u.username} for u in User.query.filter_by(role='doctor').all()]
        return render_template('admin_dashboard.html', username=session['username'], patientss=patientss, doctorss=doctorss)
    return redirect(url_for('home'))


@app.route('/admin_delete_user', methods=['POST'])
def admin_delete_user():
    if 'username' in session and session['user_type'] == 'admin':
        user_type = request.form.get('user_type')
        username_to_delete = request.form.get('username')
        if user_type and username_to_delete:
            u = User.query.filter_by(username=username_to_delete, role=user_type).first()
            if u:
                db.session.delete(u)
                db.session.commit()
    return redirect(url_for('admin_dashboard'))


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
            if User.query.filter_by(username=username).first():
                return render_template('register.html', message='Username already exists. Try another one.')
            db.session.add(User(username=username, password=password, role=user_type))
            db.session.commit()
            return redirect(url_for('home'))
        return render_template('register.html', message='Username, password, and user type are required.')
    return render_template('register.html')


def user_exists(username):
    return User.query.filter_by(username=username).first() is not None


@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    if request.method == 'POST':
        patient_username = session.get('username')
        doctor_username = request.form.get('doctor')
        date = request.form.get('date')
        time_ = request.form.get('time')
        age = request.form.get('age')
        phone = request.form.get('phone')
        session_purpose = request.form.get('session_purpose')
        if patient_username and doctor_username and date and time_:
            appt = Appointment(doctor=doctor_username, patient=patient_username, date=date, time=time_, age=age, phone=phone, session_purpose=session_purpose, status='pending')
            db.session.add(appt)
            db.session.commit()
            return redirect(url_for('dashboard'))
        return render_template('patient_dashboard.html', username=session['username'], message='All fields are required.')


def save_appointment_to_csv(*args, **kwargs):  # legacy no-op placeholder
    pass


def get_doctor_appointments(doctor_username):
    appts = Appointment.query.filter_by(doctor=doctor_username).filter(Appointment.status=='pending').all()
    return [{
        'id': a.id,
        'doctor': a.doctor,
        'patient': a.patient,
        'date': a.date,
        'time': a.time,
        'age': a.age,
        'phone': a.phone,
        'session_purpose': a.session_purpose,
        'status': a.status
    } for a in appts]


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
    appts = Appointment.query.filter_by(doctor=doctor_username, status='accept').all()
    return [{
        'id': a.id,
        'doctor': a.doctor,
        'patient': a.patient,
        'date': a.date,
        'time': a.time,
        'age': a.age,
        'phone': a.phone,
        'session_purpose': a.session_purpose,
        'status': a.status
    } for a in appts]


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
        appointments_df = pd.read_csv(patient_appointment_csv_file_path)
        appointments_df = appointments_df[~((appointments_df['doctor'] == doctor_username) &
                                           (appointments_df['condition'] == 'accept') &
                                           (appointments_df['patient'] == appointment['patient']) &
                                           (appointments_df['date'] == appointment['date']) &
                                           (appointments_df['time'] == appointment['time']))]
        appointments_df.to_csv(patient_appointment_csv_file_path, index=False)
    except FileNotFoundError:
        pass


def remove_appointments_from_csv(*args, **kwargs):  # legacy placeholder
    pass


@app.route('/accept_appointment/<int:appointment_id>', methods=['POST'])
def accept_appointment(appointment_id):
    if 'username' in session and session['user_type'] == 'doctor':
        action = request.form.get('action')
        appt = Appointment.query.filter_by(id=appointment_id).first()
        if appt and appt.doctor == session['username'] and appt.status == 'pending':
            if action == 'accept':
                appt.status = 'accept'
            elif action == 'reject':
                appt.status = 'reject'
            db.session.commit()
        return redirect(url_for('doctor_dashboard'))
    return redirect(url_for('home'))


def save_accepted_appointment(*args, **kwargs):  # legacy placeholder
    pass

@app.route('/complete_appointment/<int:appointment_id>', methods=['POST'])
def complete_appointment(appointment_id):
    if 'username' in session and session['user_type'] == 'doctor':
        appt = Appointment.query.filter_by(id=appointment_id, doctor=session['username']).first()
        if appt and appt.status == 'accept':
            appt.status = 'completed'
            db.session.commit()
    return redirect(url_for('doctor_dashboard'))

warnings.filterwarnings("default")
if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8080'))
    socketio.run(app, host=host, port=port)
