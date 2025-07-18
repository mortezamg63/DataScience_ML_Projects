This project is a language detection tool deployed as a web application using Docker. It leverages a machine learning model that predicts the language of a given text. The backend is built with FastAPI, exposing an API endpoint for language prediction, where the input text is preprocessed and passed to the model for language detection. The model is loaded from a serialized pickle file, and the detected language is returned as the response. The frontend is a simple HTML page where users can input text, and upon submission, the page sends the text to the FastAPI API, displaying the predicted language. Docker is used to containerize the application, ensuring consistent and scalable deployment of the web service.

The output's screenshot

<img width="503" height="188" alt="output_screenshot" src="https://github.com/user-attachments/assets/9de5abe2-eb74-4001-b435-9a2027b8f089" />

