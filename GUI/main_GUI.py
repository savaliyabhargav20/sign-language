import customtkinter
import cv2
from PIL import Image, ImageTk
import os
from customtkinter import CTkImage
import tkinter as tk
import tensorflow as tf
import mediapipe as mp
import numpy as np

# Load your trained model
model = tf.keras.models.load_model('sign_language_model.keras')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to extract keypoints from the video frame
def extract_keypoints_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    keypoints = []
    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks[0].landmark:
            keypoints.append(landmark.x)
            keypoints.append(landmark.y)
            keypoints.append(landmark.z)
    return keypoints if len(keypoints) == 63 else None

def toggle():
    T_frame = customtkinter.CTkFrame(screen, width=350, height=screen_height,border_width=3,border_color="#7DF9FF")
    T_frame.place(x=0, y=0)

    def close():
        T_frame.destroy()

    def toggle_theme():
        if theme_switch_var.get():
            customtkinter.set_appearance_mode("Dark")
            screen.configure(bg="black")
            frame1.configure(fg_color="#242424")
            frame2.configure(fg_color="#242424")
            frame3.configure(fg_color="#242424")
            frame4.configure(fg_color="#242424")
            sub_frame_2.configure(fg_color="#242424")
            image_frame.configure(fg_color="#242424")
        else:
            customtkinter.set_appearance_mode("Light")
            screen.configure(bg="white")
            frame1.configure(fg_color="#ebebeb")
            frame2.configure(fg_color="#ebebeb")
            frame3.configure(fg_color="#ebebeb")
            frame4.configure(fg_color="#ebebeb")
            sub_frame_2.configure(fg_color="#ebebeb")
            image_frame.configure(fg_color="#ebebeb")

    light_label = customtkinter.CTkLabel(T_frame, text="Light", font=("helvetica", 16))
    light_label.place(x=75, y=103)

    theme_switch_var = customtkinter.BooleanVar()
    theme_switch = customtkinter.CTkSwitch(T_frame, text="Dark", variable=theme_switch_var, command=toggle_theme, font=("helvetica", 16), switch_height=30, switch_width=65)
    theme_switch.place(x=115, y=103)

    toggle_btn = customtkinter.CTkButton(T_frame, text="☰", width=45, height=40, font=("helvetica", 20), command=close, text_color="black")
    toggle_btn.place(x=5, y=5)

    # Remember toggle state (Light/Dark)
    theme_switch_var.set(customtkinter.get_appearance_mode() == "Dark")

# Initialize and set the theme
customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme("green")

screen = customtkinter.CTk()
screen_width = screen.winfo_screenwidth()
screen_height = screen.winfo_screenheight()
screen.geometry(f"{screen_width}x{screen_height}+0+0")
screen.title("Sign Language To Text Conversion")

frame1 = customtkinter.CTkFrame(screen, width=screen_width, height=50,fg_color="#ebebeb")
frame1.place(x=0, y=0)

title = customtkinter.CTkLabel(frame1, text="Sign Language To Text Conversion", font=("helvetica", 36),text_color="#1DE9B6")
title.place(x=500, y=5)

toggle_btn = customtkinter.CTkButton(frame1, text="☰", width=45, height=40, font=("helvetica", 20), command=toggle, text_color="black")
toggle_btn.place(x=5, y=5)

frame2 = customtkinter.CTkFrame(screen, width=900, height=360,fg_color="#ebebeb")
frame2.place(x=10, y=55)

image_frame = customtkinter.CTkFrame(frame2, width=665, height=350, corner_radius=10,border_color="#9400D3",border_width=4.5,fg_color="#ebebeb")
image_frame.place(x=225, y=5)

def show_img(no):
    img_path = f"D:/Sign Lang. to Text project/GUI/saved_img/{no}.jpeg"
    if os.path.exists(img_path):
        img = Image.open(img_path)
        img = img.resize((655, 340))
        ctk_img = CTkImage(light_image=img, dark_image=img, size=(655, 340))
        
        for widget in image_frame.winfo_children():
            widget.destroy()
        
        label = customtkinter.CTkLabel(image_frame, text=" ", image=ctk_img)
        label.place(x=5,y=5)
    else:
        print(f"Error: Image {img_path} not found.")

sub_frame_2 = customtkinter.CTkFrame(frame2,height=300,border_color="#1DE9B6",fg_color="#ebebeb",border_width=2)
sub_frame_2.place(x=5,y=50)

def alphabet():
    for widget in sub_frame_2.winfo_children():
        widget.destroy()
    
    alphabets=["A", "B", "C", "D", "E", "F", "G", "H",
           "I","J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"] # List of actions

    for i, letter in enumerate(alphabets):
        letter_button = customtkinter.CTkButton(sub_frame_2, text=letter,
            anchor="center",
            font=("Helvetica", 14,"bold"),
            width=35,
            height=25,
            command=lambda l=letter: show_img(l))
        
        letter_button.grid(row=i // 3,column=i % 3,padx=5,pady=5) 

alphabet_btn = customtkinter.CTkButton(frame2,text="Alphabets",height=30,width=100,command=alphabet)
alphabet_btn.place(x=10,y=10)

def number():
    for widget in sub_frame_2.winfo_children():
        widget.destroy()

    numbers=[str(i) for i in range(10)]
    for i,no in enumerate(numbers):
        number_button = customtkinter.CTkButton(sub_frame_2,text=no,
            anchor="center",
            font=("Helvetica",14,"bold"),
            width=35,
            height=25,
            command=lambda l=no: show_img(l))
        
        number_button.grid(row=i // 2,column=i % 2,padx=(10),pady=(10))

number_btn = customtkinter.CTkButton(frame2,text="Numbers",height=(30),width=(100),command=(number))
number_btn.place(x=(115),y=(10))

frame3 = customtkinter.CTkFrame(screen,width=(755),height=(360),)#g_color="#ebebeb"
frame3.place(x=(10),y=(425))

output_label = customtkinter.CTkLabel(frame3,text="",font=("Helvetica",54),anchor="center")
output_label.pack()

frame4 = customtkinter.CTkFrame(screen,width=(600),height=(450),fg_color="#ebebeb",border_color="#FF6EC7",border_width=3)
frame4.place(x=(915),y=(55))

camera_label = customtkinter.CTkLabel(frame4,text=" ")
camera_label.place(x=(50),y=(10))

camera = None

actions = ["0", "1", "2", "3", "4","5", "6", "7", "8", "9","A", "B", "C", "D", "E", "F", "G", "H",
        "I","J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"] # List of actions

def start_camera():
    global camera,camera_label
    if camera is None:
        camera=cv2.VideoCapture(0)
    
    if camera.isOpened():
        update_camera_feed()

def update_camera_feed():
    global camera,camera_label
    if camera and camera.isOpened():
        ret, frame = camera.read()
        
        if ret:
            keypoints = extract_keypoints_from_frame(frame)
            
            if keypoints is not None:
                keypoints_array = np.array(keypoints).flatten()
                
                # Pad or truncate keypoints_array to match expected size (21) for one hand
                if len(keypoints_array) < 21:
                    keypoints_array = np.pad(keypoints_array,(0,(21 - len(keypoints_array))),'constant')
                else:
                    keypoints_array = keypoints_array[:21]
                
                # Reshape the array to match input shape (1, 1, 21) for the model
                keypoints_array = keypoints_array.reshape(1, 1, 21)

                # Make prediction with your model
                prediction=model.predict(keypoints_array)
                predicted_label_index=np.argmax(prediction)
                predicted_label=actions[predicted_label_index]
                output_label.configure(text=f"prediction : {predicted_label}")

                # Log the prediction probabilities for debugging
                print(f"Predicted probabilities: {prediction}")

            img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            camera_label.configure(image=img_tk)
            camera_label.image = img_tk
            
            camera_label.after(10,update_camera_feed)

def stop_camera():
    global camera
    if camera:
        camera.release()
        camera=None
    camera_label.configure(image='')

start_camera_btn = customtkinter.CTkButton(frame4,text="Start Camera",command=(start_camera))
start_camera_btn.place(x=(130),y=(405))

stop_camera_btn = customtkinter.CTkButton(frame4,text="Stop Camera",command=stop_camera)
stop_camera_btn.place(x=(330),y=(405))

frame3 = customtkinter.CTkFrame(screen,width=(755),height=(360),fg_color="#ebebeb")
frame3.place(x=(10),y=(425))

note_1=customtkinter.CTkLabel(frame3,font=("new times roman",22),text_color="#FF073A",text="**Use Right hand for Number** \n  **Use Left hand for Alphabet** ")
note_1.place(x=20 ,y=45)

output_label = customtkinter.CTkLabel(frame3,text=" ",font=("Helvetica",54),anchor="center",text_color="#FF00FF")
output_label.place(x=100,y=200)

def close_camera():
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
    
    screen.destroy()

screen.protocol("WM_DELETE_WINDOW", close_camera)
screen.mainloop()