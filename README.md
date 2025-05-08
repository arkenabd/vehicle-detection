Simple Vehicle Object Detection Web Application
-----------------------------------------------

Project Description
-------------------
This project is a simple implementation of a web application to detect vehicle objects in images using a pre-trained Deep Learning model from TensorFlow Hub and the Flask web framework in Python. Users can upload images through the web interface, and the application will display the detected images with bounding boxes and a list of detected objects.

Key Features
------------
- Upload images through the web interface.
- Run object detection inference using a pre-trained Deep Learning model.
- Display the detected images with bounding boxes and labels.
- Present a list of detected objects along with confidence scores.

Prerequisites
-------------
- Python 3.6+: You can download it from python.org.
- pip: Usually installed with Python. Virtual Environment (optional, but highly recommended): Such as venv (built-in Python) or conda.

Installation
------------
Follow these steps to get this project up and running on your local environment:
pip install Flask tensorflow tensorflow-hub opencv-python matplotlib numpy

File Structure
--------------
The project structure is expected to look like this:

.
├── app.py # Flask backend code
├── templates/ # Folder for HTML templates
│ └── index.html # Frontend templates
└── static/ # Folder for storing output images (will be created automatically)

Make sure the index.html file is inside the templates folder and the app.py file is in the root directory of the project. The static folder will be created automatically when the application is run.

Usage
-----
Run the Flask App:
Open a terminal or Command Prompt, navigate to your project’s root directory (where app.py is located), make sure the virtual environment is active, and then run the command:
python app.py

You should see output in the terminal indicating that the Flask server is running and an accessible address (usually http://127.0.0.1:5000/).

Access the App
--------------
1. Open your web browser and navigate to the address listed in the terminal output (for example, http://127.0.0.1:5000/).
2. Upload an Image:
   On the web page that opens, click the button to choose a file, select an image containing a vehicle from your computer, and then click the “Detect Objects” button.
3. The app will process the image, run detection, and display the resulting image with bounding boxes and a list of detected objects on the same page.

Deep Learning Model
-------------------
The app uses the SSD MobileNet V2 FPNLite object detection model from TensorFlow Hub. This model is a pre-trained model trained on a large dataset like COCO, which covers a wide range of object categories including vehicles. Using a pre-trained model allows us to directly perform inference without having to train the model from scratch.

Dataset
--------
Although this application focuses on inference using a pre-trained model, annotated datasets play a vital role in developing Computer Vision models. An example of a relevant dataset is the “Vehicle Detection Image Dataset” available on Kaggle (https://www.kaggle.com/datasets/pkdarabi/vehicle-detection-image-dataset). Such datasets contain images with accurate bounding box annotations and labels, which are very useful for:
- Training custom object detection models.
- Fine-tuning the pre-trained model to be more specific to a particular type of data.
- Quantitatively evaluating the model performance.

Potential for Further Development
---------------------------------
This basic project can be further developed by adding features such as:
- Detection on videos or live streams from webcams.
- Adding more object categories to detect.
- Use different object detection models or train custom models.
- Improve user interface (UI) and user experience (UX).
