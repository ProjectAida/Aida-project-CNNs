# Installing Anaconda, Jupyter Notebook, and create a Python 3.7 Environment for this project

## Installing Anaconda

1. **Download Anaconda Installer**
   - Visit the [Anaconda download page](https://www.anaconda.com/products/individual).
   - Choose the version for your operating system (Windows, macOS, or Linux).

2. **Run the Installer**
   - **Windows**: Double-click the `.exe` file.
   - **macOS**: Open the `.pkg` file.
   - **Linux**: Run `bash <Anaconda-installer-script.sh>` in the terminal.

3. **Complete the Installation**
   - Follow the prompts. Select "Add Anaconda to my PATH environment variable" for ease of use.

## Installing Jupyter Notebook

1. **Open Anaconda Navigator**
   - Find Anaconda Navigator in your applications.

2. **Launch Jupyter Notebook**
   - Click 'Launch' under Jupyter Notebook in Anaconda Navigator.

## Creating a Python 3.7 Environment

1. **Open Anaconda Prompt/Terminal**
   - **Windows**: Use Anaconda Prompt.
   - **macOS/Linux**: Use the terminal.

2. **Create a New Environment**
   - Run `conda create -n myenv python=3.7`, where myenv is the name of your new environment..

3. **Activate the Environment**
   - Run `conda activate myenv`.

## Installing Dependencies from `requirements.txt`

4. **Install Packages from `requirements.txt`**
   - Activate your environment (`conda activate myenv`).
   - Navigate to your `requirements.txt` file (`cd /path/to/directory`).
   - Run `pip install -r requirements.txt`.

# License
This project is part of the Aida project (projectaida.org) for which license information can be found on https://github.com/ProjectAida/aida. 

# References
   - Liu, Yi, Leen-Kiat Soh, and Elizabeth Lorang. "Investigating coupling preprocessing with shallow and deep convolutional neural networks in document image classification." Journal of Electronic Imaging 30, no. 4 (2021): 043024-043024.
   - Soh, Leen-Kiat, Elizabeth Lorang, and Yi Liu. "Aida: intelligent image analysis to automatically detect poems in digital archives of historic newspapers." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1. 2018.


Code in files, datagen_image.py and model_factory.py, is created by Yi Liu (email:yil@unl.edu) at Aida team at University of Nebraska-Lincoln. All rights are reserved. 

