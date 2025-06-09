import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import smtplib
from email.message import EmailMessage
import mediapipe as mp
import csv
import copy
import itertools
from dotenv import load_dotenv
import os
import tensorflow as tf
from db import create_user_table, add_user, authenticate_user

# Initialize DB
create_user_table()

# Import custom modules
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier
from model.utils import calc_bounding_rect, calc_landmark_list, pre_process_landmark, pre_process_point_history

# Email settings
load_dotenv()

def send_email(subject, body, to="EMAIL_ADDRESS"):
    sender = os.getenv("EMAIL_ADDRESS")
    password = os.getenv("EMAIL_PASSWORD")

    msg = EmailMessage()
    msg.set_content(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, password)
        server.send_message(msg)

# Dummy user database
if "users" not in st.session_state:
    st.session_state.users = {"admin": "admin123"}

# Page configs
st.set_page_config(page_title="Hand Gesture Recognition", layout="wide")

# Authentication system
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Sidebar navigation
pages = ["Home", "About Us", "User Manual", "Recognition Model", "Contact Us"]
selected_page = st.sidebar.selectbox("Navigation", pages)

# User account icon and username display
if st.session_state.logged_in:
    username = st.session_state.username
    st.sidebar.markdown(f"### Welcome, {username}")
    st.sidebar.image("https://www.gravatar.com/avatar/00000000000000000000000000000000?s=200&d=mp", width=50)  # Placeholder icon

# Logout button if logged in
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.sidebar.success("Logged out successfully.")


# --- Pages ---

if selected_page == "Home":
    st.title("Welcome to the Hand Gesture Recognition System")
    st.markdown("<h3 style='text-align: center;'>Explore the World of Hand Signs</h3>", unsafe_allow_html=True)

    # Row 1: Hello Gesture
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("images/image_1.jpeg", use_container_width=True)
    with col2:
        st.markdown(
            """
            <div style="text-align: center;">
                <p>Hand sign language is a visual form of communication that uses hand gestures, body movements, and facial expressions to convey words and meanings. It is primarily used by deaf and hard-of-hearing individuals as an alternative to spoken language. Each gesture or combination of gestures represents specific letters, words, or concepts.</p>
                <p>There are various types of sign languages used across the world, such as American Sign Language (ASL), British Sign Language (BSL), and Indian Sign Language (ISL), each with its own grammar and vocabulary. Hand sign language is not universal—different regions have developed their own versions based on culture and linguistic needs.</p>
                <p>With the advancement of technology, hand sign language recognition systems are being developed using computer vision and machine learning to help bridge the communication gap between hearing and non-hearing individuals. These systems translate signs into text or speech, enabling more inclusive communication.</p>
            </div>
            """, unsafe_allow_html=True
        )

    st.markdown("---")

    # Row 2: Thank You Gesture (Image on Right)
    col3, col4 = st.columns([2, 1])  # Swapped the width ratio
    with col3:
        st.markdown(
            """
            <div style="text-align: center;">
                <p>Hand sign languages are crucial for enabling effective communication within the deaf and hard-of-hearing community. For many individuals, these languages serve as the primary means of expression, offering a way to convey thoughts, emotions, and ideas. Without sign language, many deaf people would face significant barriers to communication, limiting their independence and ability to fully participate in society. Sign language empowers individuals by providing a natural form of communication that does not rely on spoken words or hearing, ensuring that they can engage in conversations and share experiences in their daily lives.</p>
                <p>Real-time recognition systems, powered by machine learning and computer vision, are breaking down barriers by enabling communication between deaf and hearing people without the need for interpreters. These innovations in assistive technology make it easier for deaf individuals to navigate the world, enhancing their ability to participate in education, work, and social activities. As technology continues to develop, the accessibility and integration of sign language will only improve, leading to greater equity and inclusion for the deaf community.</p>
            </div>
            """, unsafe_allow_html=True
        )
    with col4:
        st.image("images/image_2.jpeg", use_container_width=True)

    st.markdown("---")

    # Row 3: Help Gesture
    col5, col6 = st.columns([1, 2])
    with col5:
        st.image("images/image_3.jpeg", use_container_width=True)
    with col6:
        st.markdown(
            """
            <div style="text-align: center;">
                <p>Hand sign language is used in various real-life scenarios, greatly enhancing communication for the deaf and hard-of-hearing individuals. In education, sign language is often taught in schools specifically designed for the deaf, providing a rich learning environment. Deaf students use sign language to interact with their peers and teachers, facilitating better understanding and participation in class. Additionally, many mainstream schools are integrating sign language programs to promote inclusivity, ensuring that deaf students have equal access to education.</p>
                <p>Furthermore, hand sign language is becoming increasingly integrated into technology to support communication in real-time. Mobile apps and smart devices are being developed to recognize and translate sign language gestures, enabling seamless communication between deaf and hearing people. For example, applications like SignAll and MotionSavvy are using computer vision and machine learning to translate hand gestures into text or speech, offering new ways for deaf individuals to interact with people who don’t know sign language, thereby enhancing inclusivity in public spaces, healthcare, and everyday social interactions.</p>
            </div>
            """, unsafe_allow_html=True
        )

elif selected_page == "About Us":
    st.markdown("<h1 style='text-align: center;'>About Us</h1>", unsafe_allow_html=True)

    # Paragraph about teamwork and dedication
    st.markdown(
        """
        <div style="text-align: center;">
            <p>Our team is driven by a shared vision of bridging the communication gap between deaf and hearing communities. We are committed to ensuring that individuals with hearing impairments have equal access to communication and information. Our diverse group of passionate individuals brings unique skills and perspectives to the table, including expertise in machine learning, computer vision, software development, and user experience design. Together, we are dedicated to developing innovative solutions that empower the deaf and hard-of-hearing community.</p>
            <p>Our goal is to create technologies that foster inclusivity and create a world where everyone can communicate effortlessly, regardless of their hearing abilities. We believe in the power of collaboration and continuous learning, which drives us to push the boundaries of what can be achieved with gesture recognition and artificial intelligence.</p>
        </div>
        """, unsafe_allow_html=True
    )

    st.markdown("---")

    
    # Team members images with names and roles
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image("images/team_member_1.jpeg", use_container_width=True)
        st.markdown("<p style='text-align: center;'>Atharva Dagade</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Project Lead</p>", unsafe_allow_html=True)
        
    with col2:
        st.image("images/team_member_2.jpeg", use_container_width=True)
        st.markdown("<p style='text-align: center;'>Siddharth Patwardhan</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>UI Creator</p>", unsafe_allow_html=True)
        
    with col3:
        st.image("images/team_member_3.jpeg", use_container_width=True)
        st.markdown("<p style='text-align: center;'>Deva Vishwakarma</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Data and Documentation Manager</p>", unsafe_allow_html=True)
        
    with col4:
        st.image("images/team_member_4.png", use_container_width=True)
        st.markdown("<p style='text-align: center;'>Dr. V. S. Inamdar</p>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Project Guide</p>", unsafe_allow_html=True)

    st.markdown("---")

    # Paragraph about the project
    st.markdown(
        """
        <div style="text-align: center;">
            <p>Our current project is focused on building a real-time hand gesture recognition system. Using cutting-edge machine learning techniques and computer vision, we have developed a system that can recognize various hand gestures in real-time through a webcam. The system is designed to bridge the communication gap between deaf and hearing individuals, enabling seamless interaction without the need for traditional sign language interpreters. We have made significant progress in creating a user-friendly interface that supports communication through gestures, and we are constantly refining the system to improve its accuracy and accessibility.</p>
            <p>Our project is not just about technology; it’s about making a difference in the lives of individuals who face challenges due to hearing impairments. We are excited to continue working towards a future where communication is universally accessible and inclusive.</p>
        </div>
        """, unsafe_allow_html=True
    )

elif selected_page == "User Manual":
    st.markdown("<h1 style='text-align: center;'>User Manual</h1>", unsafe_allow_html=True)
    st.markdown("------")
    st.markdown("""
    <div style='text-align: center; font-size: 18px;'>
        <p><strong>1. Explore gestures</strong> by scrolling through the grid of hand sign images and their names.</p>
        <p><strong>2. Identify each gesture</strong> by reading the name below the corresponding image.</p>
        <p><strong>3. Navigate the grid</strong> to easily browse through all available hand signs and their meanings.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("------")

    image_folder = "sign_images"  # make sure this folder contains 55 images
    gesture_names = [
       "Down", "Eat", "Fast", "Father", "Favourite", "Friend", "Go", "Hate", "Hello", "How",
    "Hurt", "I_agree", "Is", "Learn", "Love", "Like", "Me", "Name", "No", "Nothing",
    "Ok", "Prosper", "Photo", "Please", "Question", "Ready", "Rock", "Run", "Shocker",
    "Sit_down", "Slow", "Stand_up", "Stop", "Teacher", "That", "This", "Together", "Up",
    "Wash", "Whats_Up", "Win", "You", "Again", "Baby", "Bird", "Busy", "Call", "Claw",
    "Come", "Dont_like"
    ]  # Replace with actual names if needed

    for row in range(11):
        cols = st.columns(5)
        for col_idx in range(5):
            sign_index = row * 5 + col_idx
            if sign_index < len(gesture_names):
                img_path = os.path.join(image_folder, f"sign_{sign_index+1}.png")
                try:
                    cols[col_idx].image(img_path, use_container_width=True)
                    cols[col_idx].markdown(f"<center>{gesture_names[sign_index]}</center>", unsafe_allow_html=True)
                except:
                    cols[col_idx].warning("Image not found.")
            
            # Add vertical separator lines after each column (except the last one)
            if col_idx < 4:
                cols[col_idx].markdown("<div style='border-right: 1px solid #ccc; height: 100%;'></div>", unsafe_allow_html=True)

        # Add a horizontal rule after every row
        st.markdown("----")



elif selected_page == "Recognition Model":
    st.markdown("<h1 style='text-align: center;'>Real-Time Gesture Recognition</h1>", unsafe_allow_html=True)
    
#     @st.cache_data
#     def load_csv_from_drive(drive_url):
#         import pandas as pd
#         return pd.read_csv(drive_url, header=None).iloc[:, 0].tolist()

# # Google Drive direct download link (you shared this)
#     csv_url = "https://drive.google.com/uc?export=download&id=1OQ5Tsn7m7YwyOyosqo_afZkJ8H_AqucU"
#     keypoint_classifier_labels = load_csv_from_drive(csv_url)

    
    if not st.session_state.logged_in:
        st.warning("Please log in to access the recognition model.")
    else:
        st.subheader("Choose Gesture Type")
        model_type = st.radio("Model Type", ["Static Gestures", "Dynamic Gestures"])

        start = st.button("Start Recognition")

        if start:
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
            mp_draw = mp.solutions.drawing_utils

            keypoint_classifier = KeyPointClassifier()
            point_history_classifier = PointHistoryClassifier()

            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            stop = st.button("Stop Recognition")

            point_history = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                debug_image = copy.deepcopy(frame)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                gesture_text = "No hand detected"
                accuracy = ""

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_draw.draw_landmarks(debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                        if model_type == "Static Gestures":
                            pre_processed_landmark_list = pre_process_landmark(landmark_list)
                            hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                            with open("model/keypoint_classifier/keypoint_classifier_label.csv", encoding='utf-8-sig') as f:
                                keypoint_classifier_labels = [row[0] for row in csv.reader(f)]
                            gesture_text = keypoint_classifier_labels[hand_sign_id]
                            

                        elif model_type == "Dynamic Gestures":
                            cx, cy = landmark_list[8]  # Index fingertip
                            point_history.append((cx, cy))
                            if len(point_history) > 16:
                                point_history = point_history[-16:]

                                pre_processed_point_history = pre_process_point_history(point_history)
                                history_id = point_history_classifier(pre_processed_point_history)
                                with open("model/point_history_classifier/point_history_classifier_label.csv", encoding='utf-8-sig') as f:
                                    point_history_labels = [row[0] for row in csv.reader(f)]
                                gesture_text = point_history_labels[history_id]
                                

                cv2.putText(debug_image, str(gesture_text), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(debug_image, str(accuracy), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                frame_rgb = cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                stframe.image(img, channels="RGB")

                if stop:
                    break

            cap.release()
            st.success("Recognition stopped.")

elif selected_page == "Contact Us":
    st.markdown("<h1 style='text-align: center;'>Contact Us</h1>", unsafe_allow_html=True)
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message")
    if st.button("Send Message"):
        if name and email and message:
            send_email("New Contact Form Submission",
                       f"Name: {name}\nEmail: {email}\nMessage: {message}",
                       "youremail@example.com")
            st.success("Message sent!")
        else:
            st.error("Please fill all fields.")

# Login/Register section (shown in sidebar only if not logged in)
if not st.session_state.get("logged_in", False):
    st.sidebar.title("Login/Register")
    form_type = st.sidebar.radio("Choose", ["Login", "Register"])

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if form_type == "Login":
        if st.sidebar.button("Login"):
            # Authenticate user from DB
            user = authenticate_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.sidebar.success("Logged in successfully!")
            else:
                st.sidebar.error("Invalid credentials.")
    
    elif form_type == "Register":
        if st.sidebar.button("Register"):
            # Check if user already exists
            user = authenticate_user(username, password)  # Checking if user already exists
            if user:
                st.sidebar.error("Username already exists.")
            else:
                # Register the new user in the database
                add_user(username, password, "")  # You can add email or other data if necessary
                st.sidebar.success("Registered successfully. Please login.")
