# ASL Sign Language Recognition App

## Overview

This project is a deep learning-based system for recognizing American Sign Language (ASL) signs from 28×28 grayscale images. The application provides an easy to use GUI and 3 DNN models to learn a dataset. The application also provides live webcam input recognition as well as recognition for a customised input dataset.

## The Application
### Dataset Import
We provide users with an efficient and easy to use tab to import data to train their desired models upon.

![image](https://github.com/user-attachments/assets/ea0c2fa2-8355-4203-ad14-80140e22e65e)

### The Dataset Viewer
Moreover, we provide an easy way for users to navigate and view their .csv file as images while also providing an efficient way to filter the dataset based on the label. We also provide basic statistics to 
visualise the number of samples. We make monitoring the dataset easy.

![image](https://github.com/user-attachments/assets/617bed7e-3b3a-493f-9685-8329eeec6fc7)
![image](https://github.com/user-attachments/assets/8098a29f-0fe4-4a9c-a466-0d1308aaadfc)

### The Training Tab
This is where the magic happens. Users can choose the model they wish to train and the hyperparameters alongside it. We provide 2 pretrained models Resnet18 and Alexnet and a 3rd custom CNN model inspired by the Inception model with SE Attention Blocks. The king of our application! You can also keep tabs on the training progress and see important metrics passed by the model as a callback, these include the validation accuracy, the training loss, the number of epochs left and timer to show the progress of the training.

![image](https://github.com/user-attachments/assets/4fb7cd0d-1161-4b83-a2de-d0e426b3de55)

### The Prediction Tab
Users can select a model that they have trained (these are the .pt files located in /saved_models) and use the model to make inferences on a dataset. Users can also turn on their webcam to take screenshots of hand signs they make!

![image](https://github.com/user-attachments/assets/be7fdd02-f957-4230-9e07-b04e27c338ae)
![image](https://github.com/user-attachments/assets/bb5e270f-3207-4cb3-9b97-207f1a07f8ab)


## The Directory
```
├── backend/  #Contains all the threads to make GUI concurrent and lag-free
│   ├── dataset.py
│   ├── dataset_loader.py
│   ├── import_thread.py
│   ├── training_thread.py
│   └── webcam_thread.py
│
├── data/ #Folder that stores the dataset imports
├── datasets/ #Provides 2 datasets - 1 training and 1 test set.
│
├── frontend/ #Contains all things GUI related
│   ├── styles/
│   ├── utils/
│   ├── dataset_import_tab.py
│   ├── dataset_viewer_tab.py
│   ├── main_window.py
│   ├── prediction_popup.py
│   ├── prediction_tab.py
│   ├── timer_widget.py
│   ├── timer.py
│   ├── training_tab.py
│   └── webcam_popup.py
│
├── ml/ # Contains all things Machine Learning related
│   ├── models/ # Architecture for the 3 DNN Models provided
│      ├── resnet.py
│      ├── lebron.py
│      ├── alexnet.py
│   ├── config.py
│   ├── evaluate.py # Script to test the saved model accuracy against provided test set.
│   ├── trainer.py # Main training loop
│   └── transforms.py
│
├── saved_models/ #Models are saved in here once training is finished.
│
├── main.py #Main entry point for the app.
├── requirements.txt
├── README.md
├── .gitignore
├── style.qss
└── uploadComplete.png
```

## Getting Started

Follow these steps to set up and run the application after cloning the repository:

### 1. Clone the Repository

```bash
git clone https://github.com/COMPSYS302/project-deep-neural-network-python-dnn_team_28.git
cd project-deep-neural-network-python-dnn_team_28
```

### 2. (Optional but Recommended) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # On macOS/Linux
venv\Scripts\activate           # On Windows
```
### 2.1 Conda Setup (Optional)

If you're using Anaconda, follow these steps instead of `python -m venv`:

1. Create a new conda environment:

```bash
conda create -n asl-env python=3.10
conda activate asl-env
```

2. Install core dependencies with `conda`:

```bash
conda install numpy pandas matplotlib opencv scikit-learn pyqt sympy tqdm -c conda-forge
```

3. Install PyTorch (CUDA 12.1):

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

4. Install remaining dependencies with pip:

```bash
pip install -r requirements.txt --no-deps
```
### 3. Install Dependencies (Skip this if you followed 2.1)

This application requires the GPU-enabled versions of `torch`, `torchvision`, and `torchaudio` built for CUDA 12.1.

To install them correctly, run the following **before** installing the rest of the requirements:

#### Windows (CMD or PowerShell):

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

#### macOS/Linux (bash/zsh):

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```
Then install the remaining packages:

```bash
pip install -r requirements.txt
```
### 4. Prepare Your Dataset
If you have custom datasets:
Ensure you have a CSV-formatted dataset where:
- The first column is the label.
- The next 784 columns are pixel values for a 28×28 grayscale image.

Place it inside the `datasets/` folder.

Otherwise, feel free to use the ones we have provided!

### 5. Run the Application (GUI)

```bash
python main.py
```

This will launch the PyQt5 GUI!

###Extra
If you wish to test the model and it's accuracy, we have provided an evaluate.py script that checks a selected model for you.

1) First head into evaluate.py
2) Change these parameters
![image](https://github.com/user-attachments/assets/dc90db96-cd39-4114-82b5-aa92cf19fda3)
3)execute the script from the command line
```bash
python -m ml.evaluate
```

### Contributors
- Dhruv Sawant - Machine Learning and Admin
- Harry Ma - Full Stack
- Leo Chu - Frontend
