# Real-Time Object Recognition with CIFAR-100 and Project Explanation App

This project combines a powerful deep learning model for object recognition with a modern web application to explain its inner workings. It consists of two main components:

1.  **Python Object Recognition Model**: A Convolutional Neural Network (CNN) trained on the CIFAR-100 dataset. It can be used for training from scratch and for real-time object recognition using a computer's webcam.
2.  **React Explanation App**: A detailed, user-friendly web application that breaks down the Python project, explaining each script, the model architecture, and tools like TensorBoard for monitoring the training process.

## Project Structure

```
/
|-- explanation-app/      # React application for project explanation
|   |-- public/
|   |-- src/
|   |-- package.json
|   `-- ...
|-- model.py              # CNN model definition (SimpleCNN)
|-- train.py              # Script to train the model
|-- predict.py            # Script for real-time prediction using the webcam
|-- utils.py              # Utility functions (e.g., data loading)
|-- requirements.txt      # Python dependencies
|-- .gitignore
`-- README.md
```

## Technologies Used

*   **Backend & Model**:
    *   Python
    *   PyTorch
    *   OpenCV
    *   NumPy
    *   Matplotlib
    *   TensorBoard

*   **Frontend**:
    *   React
    *   JavaScript

## Setup and Installation

### 1. Python Environment

It is recommended to use a virtual environment to manage Python dependencies.

```bash
# Create and activate a virtual environment (e.g., venv)
python -m venv venv
# On Windows, use `venv\Scripts\activate`
# On Linux/macOS, use `source venv/bin/activate`

# Install the required packages
pip install -r requirements.txt
```

### 2. React Explanation App

Navigate to the `explanation-app` directory and install the necessary Node.js packages.

```bash
cd explanation-app
npm install
```

## Usage

### 1. Training the Model

To train the CNN model on the CIFAR-100 dataset, run the `train.py` script from the root directory.

```bash
python train.py
```

You can monitor the training process using TensorBoard:

```bash
tensorboard --logdir=runs
```

### 2. Real-Time Object Recognition

After training, a model file (e.g., `cifar100_model.pth`) will be saved. Use this model for real-time recognition with your webcam by running `predict.py`.

```bash
python predict.py
```

### 3. Viewing the Explanation App

To launch the React application that explains the project:

```bash
cd explanation-app
npm start
```

This will open a new tab in your browser with the explanation app.

## Model Architecture: `SimpleCNN`

The `SimpleCNN` model, defined in `model.py`, has the following structure:

1.  **Four Convolutional Blocks**: Each block consists of a `Conv2d` layer, `BatchNorm2d`, `ReLU` activation, and `MaxPool2d` (except for the last block).
    *   **Input**: 3x32x32 images.
    *   **Output**: A feature map of size 256x4x4.
2.  **Flatten Layer**: Converts the 3D feature map into a 1D vector.
3.  **Three Fully-Connected (FC) Layers**:
    *   These layers classify the features into one of the 100 CIFAR classes.
    *   Dropout is used to prevent overfitting.

For more details, refer to the code in `model.py` or the explanation app.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
