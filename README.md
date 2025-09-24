1. Create a Virtual Environment
Open your terminal or command prompt.
Navigate to your project's root directory (GymConnect).

2.Run the following command to create a virtual environment 

python -m venv .venv

3.This will create a new folder named .venv in your project directory.

4. Activate the Virtual Environment
You need to activate the virtual environment to use it. The command depends on your operating system:

On Windows:

.venv\Scripts\activate

On macOS and Linux:

source .venv/bin/activate

Once activated, your terminal prompt will change to show the name of the virtual environment (e.g., (.venv) GymConnect).

5. Install Project Dependencies
After activating the virtual environment, you can install the required libraries listed in your requirements.txt file.

Make sure you are in the project's root directory.

Run the following command:

pip install -r requirements.txt

6.This will install numpy, opencv-python-headless, mediapipe, streamlit, and streamlit_webrtc specifically within this isolated environment.

7. Run Your Application within the Virtual Environment
Now that the virtual environment is active and the dependencies are installed, you can run your application as usual.

To start the Flask backend: run cmd through backend and give command

python app.py

In a new terminal (after activating the virtual environment there as well), to start the Streamlit frontend:

streamlit run "🏠️GYMCONNECT.py"
