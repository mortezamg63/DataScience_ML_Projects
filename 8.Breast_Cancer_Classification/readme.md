Here is the code explanation in **Markdown format**, summarizing what the code does:

---

## Breast Cancer Classification using CNN

This project builds and evaluates a Convolutional Neural Network (CNN) model for classifying breast cancer images from the IDC dataset. The workflow includes data loading, preprocessing, augmentation, model building, training, evaluation, and plotting performance metrics.

### 🧾 Libraries and Setup
- Imports essential libraries: `Keras`, `TensorFlow`, `OpenCV`, `NumPy`, `Matplotlib`, and others for preprocessing, model building, visualization, and performance evaluation.
- Defines paths for training, validation, and test datasets, and sets hyperparameters like `EPOCHS`, `BATCH_SIZE`, and learning rate.

### 🗃️ Dataset Structure
- The dataset originally contains images stored in a directory (`datasets/original`), and a script (commented out) is available to reorganize images into training, validation, and testing folders based on defined split ratios.
  
### 🧠 CNN Model (CancerNet)
A custom CNN is defined with:
- `SeparableConv2D` layers for efficiency.
- `ReLU` activations and `BatchNormalization`.
- `MaxPooling` and `Dropout` for regularization.
- A `Flatten` layer followed by two `Dense` layers ending in `softmax` for multi-class classification.

### 🔄 Data Augmentation and Generators
- Uses `ImageDataGenerator` for augmenting training images with transformations such as rotation, zoom, shifts, shear, and flips.
- Validation and test sets are rescaled without augmentation.
- `flow_from_directory` is used to load images from directory structure into batches.

### 🚀 Model Training
- Compiles the CNN using the `Adagrad` optimizer and binary cross-entropy loss.
- Trains the model using the `fit_generator` method over the training and validation sets for 40 epochs.
- Applies `class_weight` to handle class imbalance.

### 📊 Model Evaluation
- Predicts labels on the test set.
- Prints a classification report using `classification_report` with precision, recall, and F1-score.
- Computes confusion matrix and calculates:
  - **Accuracy**
  - **Specificity**
  - **Sensitivity**

### 📈 Results Visualization
- Plots training and validation loss and accuracy over epochs.
- Saves the plot as `plot.png`.

