import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageTk


model = load_model('ms-lesion-md(6).ipynb') 
def predict_image(image_path):
    image = Image.open(image_path)
    image = image.resize((150, 150))  
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return prediction

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        prediction = predict_image(file_path)
        messagebox.showinfo("Prediction", f"IA prédit: {prediction}")

root = tk.Tk()
root.title("Interface pour modèle d'IA")


open_button = tk.Button(root, text="Ouvrir une image", command=open_image)
open_button.pack()
open_button2=tk.Button(root,text="Segmenter",command=open_image)
open_button2.pack()
root.mainloop()