import pickle
import re
from pathlib import Path

__version__ = "0.1.0" # our model version (I directly mentioned here)

#Path(__file__) returns the path of the current file to get the base directory
BASE_DIR = Path(__file__).resolve(strict=True).parent 

with open(f"{BASE_DIR}/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)


classes = ['Arabic', 'Danish', 'Dutch', 'English', 'French', 'German',
       'Greek', 'Hindi', 'Italian', 'Kannada', 'Malayalam', 'Portugeese',
       'Russian', 'Spanish', 'Sweedish', 'Tamil', 'Turkish']

def predict_pipeline(text):
    # Preprocessing the input text
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()

    # prediction
    pred = model.predict([text])
    return classes[pred[0]]