import tkinter as tk
from tkinter import filedialog, Label, Button, Frame, messagebox
from PIL import Image, ImageTk  # Pillow for image handling in Tkinter
import numpy as np
import cv2
import os
# Use the modern Keras import if available, otherwise fallback for older TensorFlow/Keras
try:
    from tensorflow.keras.models import load_model
except ImportError:
    from keras.models import load_model

# --- Configuration (derived from your script) ---
MODEL_DIR = "model"
MODEL_WEIGHTS_PATH_KERAS = os.path.join(MODEL_DIR, "cnn_weights.keras") # Prefer .keras
MODEL_WEIGHTS_PATH_HDF5 = os.path.join(MODEL_DIR, "cnn_weights.hdf5") # Fallback
IMG_WIDTH, IMG_HEIGHT = 32, 32
CLASS_LABELS = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# --- Global Variables for GUI State ---
loaded_model = None
selected_image_path = None
# Declare the GUI widget variables globally so they can be accessed by functions
image_display_label = None
result_label = None
path_display_label = None

# --- Core Logic (build_and_load_model, preprocess_image_for_prediction - unchanged) ---
def build_and_load_model():
    global loaded_model
    model_path_to_try = None
    if os.path.exists(MODEL_WEIGHTS_PATH_KERAS):
        model_path_to_try = MODEL_WEIGHTS_PATH_KERAS
    elif os.path.exists(MODEL_WEIGHTS_PATH_HDF5):
        model_path_to_try = MODEL_WEIGHTS_PATH_HDF5

    if model_path_to_try:
        try:
            loaded_model = load_model(model_path_to_try)
            print(f"Model loaded successfully from {model_path_to_try}")
            return True
        except Exception as e:
            messagebox.showerror("Model Load Error", f"Error loading model from {model_path_to_try}:\n{e}\nPlease ensure the model file exists and is compatible.")
            print(f"Error loading model: {e}")
            return False
    else:
        messagebox.showerror("Model Load Error", f"Model weights file not found at:\n{MODEL_WEIGHTS_PATH_KERAS}\nor\n{MODEL_WEIGHTS_PATH_HDF5}\nPlease ensure the model is trained and saved in the '{MODEL_DIR}' directory.")
        return False

def preprocess_image_for_prediction(image_path):
    try:
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            print(f"Error: Could not read image at {image_path}")
            return None
        img_resized = cv2.resize(img_cv, (IMG_WIDTH, IMG_HEIGHT))
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        elif img_resized.shape[2] == 1:
             img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = img_array.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        messagebox.showerror("Image Error", f"Could not process image: {e}")
        return None

# --- CORRECTED Tkinter Function for Image Selection ---
def select_image_button_action():
    """
    Handles the 'Select Image' button click.
    Opens a file dialog, loads and displays the image, keeping a reference.
    """
    # Access global variables needed
    global selected_image_path, image_display_label, result_label, path_display_label

    print("Select image button clicked.") # Debug print
    # Clear previous results and image
    if result_label:
      result_label.config(text="Prediction: ")
    if image_display_label:
      image_display_label.config(image='') # Clear image from label
      image_display_label.image = None # Clear the reference too

    file_path = filedialog.askopenfilename(
        title="Select Brain MRI Image",
        filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.tif"), ("All files", "*.*"))
    )
    print(f"File selected: {file_path}") # Debug print

    if file_path:
        selected_image_path = file_path # Store the path globally
        try:
            print("Loading image with PIL...") # Debug print
            img_pil = Image.open(selected_image_path)
            print(f"PIL Image loaded successfully: size={img_pil.size}, mode={img_pil.mode}")

            # --- Resize for display in the GUI ---
            display_max_width = 350  # Max width for the display area
            display_max_height = 350 # Max height for the display area
            img_copy = img_pil.copy() # Work on a copy for display resizing
            # Resize while maintaining aspect ratio to fit within the bounds
            img_copy.thumbnail((display_max_width, display_max_height), Image.LANCZOS) # LANCZOS is preferred for Pillow >= 9.1.0
            print(f"Image resized for display to: {img_copy.size}")

            # Convert the PIL image to a PhotoImage object for Tkinter
            print("Converting to PhotoImage...")
            tk_image_for_display = ImageTk.PhotoImage(img_copy)
            print("PhotoImage created.")

            # --- This is the crucial part ---
            # Update the label widget to show the new image
            image_display_label.config(image=tk_image_for_display)
            # Keep a reference to the PhotoImage object by attaching it to the widget.
            # If you don't do this, Python's garbage collector might remove the image,
            # causing it not to display.
            image_display_label.image = tk_image_for_display
            print("Reference to PhotoImage stored in label widget.")

            # Update path display label
            path_display_label.config(text=f"Selected: ...{os.path.basename(selected_image_path)}", wraplength=380)
            print("Path label updated.")

        except Exception as e:
            print(f"ERROR during image loading/display: {e}") # Print error to console
            messagebox.showerror("Image Load Error", f"Failed to load or display image:\n{e}")
            selected_image_path = None
            image_display_label.config(image='') # Clear image on error
            image_display_label.image = None # Clear reference
            path_display_label.config(text="Selected: None")
    else:
        # User cancelled the file dialog
        print("No file selected.")
        selected_image_path = None
        image_display_label.config(image='')
        image_display_label.image = None
        path_display_label.config(text="Selected: None")


def predict_button_action():
    """
    Handles the 'Predict Tumor Type' button click. (Unchanged)
    """
    if loaded_model is None:
        messagebox.showerror("Error", "Model not loaded. Cannot predict.")
        return
    if selected_image_path is None:
        messagebox.showinfo("No Image", "Please select an image first.")
        return

    processed_img_array = preprocess_image_for_prediction(selected_image_path)

    if processed_img_array is not None:
        try:
            prediction_probabilities = loaded_model.predict(processed_img_array)
            predicted_class_index = np.argmax(prediction_probabilities, axis=1)[0]

            if 0 <= predicted_class_index < len(CLASS_LABELS):
                predicted_label_str = CLASS_LABELS[predicted_class_index]
                result_text = f"Prediction: {predicted_label_str}"
            else:
                result_text = "Prediction: Unknown class index"
                print(f"Warning: Predicted class index {predicted_class_index} is out of bounds for labels.")

            result_label.config(text=result_text)

        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")
            result_label.config(text="Prediction: Error")

# --- Main Application Setup (Ensure labels are defined before functions use them) ---
def main_gui():
    # Define these as global so the functions above can modify them
    global image_display_label, result_label, path_display_label

    root = tk.Tk()
    root.title("Brain Tumor Classification - Deployment")
    root.geometry("500x600")
    root.configure(bg="#f0f0f0")

    # Attempt to load the model on startup
    model_loaded_successfully = build_and_load_model()

    # --- GUI Elements ---
    title_label = Label(root, text="Brain Tumor Classifier", font=("Helvetica", 18, "bold"), bg="#f0f0f0", fg="#333")
    title_label.pack(pady=(15,10))

    button_frame = Frame(root, bg="#f0f0f0")
    button_frame.pack(pady=10)

    select_button = Button(button_frame, text="Select Brain MRI Image", command=select_image_button_action, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="flat", padx=10, pady=5)
    select_button.pack(side=tk.LEFT, padx=10)

    predict_button = Button(button_frame, text="Predict Tumor Type", command=predict_button_action, font=("Helvetica", 12), bg="#007BFF", fg="white", relief="flat", padx=10, pady=5)
    predict_button.pack(side=tk.LEFT, padx=10)
    if not model_loaded_successfully:
        predict_button.config(state=tk.DISABLED)

    path_display_label = Label(root, text="Selected: None", font=("Helvetica", 9), bg="#f0f0f0", fg="#555")
    path_display_label.pack(pady=(0,10))

    # --- Image Display Label ---
    # Create the label widget where the image will be shown
    image_display_label = Label(root, bg="#dddddd", relief="sunken") # Start with a background color
    # Pack it into the window
    image_display_label.pack(pady=10, padx=20)

    # --- Result Label ---
    result_label = Label(root, text="Prediction: ", font=("Helvetica", 14, "bold"), bg="#f0f0f0", fg="#333", height=2) # Give it a bit more height
    result_label.pack(pady=(10, 20)) # Adjusted padding

    root.mainloop()

# --- Entry Point Check ---
if __name__ == "__main__":
    if not os.path.isdir(MODEL_DIR):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Setup Error", f"The directory '{MODEL_DIR}' was not found.\nPlease ensure it exists and contains the model weights.")
        root.destroy()
    else:
        main_gui()