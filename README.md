# Voice Transfer using GANs

This repository contains the code for training and generating audio using a GAN model for voice transfer. The GAN model is trained to replicate the voice of a specific singer and generate new audio that sounds like the singer's voice.

## Contents

The repository contains the following files:

- `preprocessing.py`: This file contains the code for preprocessing the audio recordings. It is used to extract the audio, Mel-frequency cepstral coefficients (MFCCs), and shifted audio from the audio recordings.

- `model.py`: This file contains the code for defining the generator and discriminator models for the GAN. The generator model is used to generate new audio that sounds like the singer's voice, and the discriminator model is used to distinguish between real and generated audio.

- `train.py`: This file contains the code for training the GAN model on the preprocessed audio data. It uses the BinaryCrossentropy loss function and the Adam optimizer to update the model parameters.

- `generate.py`: This file contains the code for generating new audio using the trained GAN model. It takes a random noise input and applies the generator model to generate new audio.

- `requirement.txt`: This file contains a list of the dependencies required to run the code in this repository.

- `main.py`: This file contains the main function that calls the different files in the right order and passes the necessary information between them.

## Prerequisites

- Python 3.6 or later
- TensorFlow 2.4 or later
- numpy 1.19.2
- librosa 0.8.0
- You will need a dataset of audio recordings of the singer whose voice you want to replicate in order to train your model.

## Getting Started

1. Clone the repository: `git clone https://github.com/yourusername/voice-transfer-gans`
2. Create a virtual environment and activate it: `python -m venv env` and `source env/bin/activate` on Linux and macOS or `env\Scripts\activate.bat` on Windows
3. Install the dependencies: `pip install -r requirements.txt`
4. Run the main function : `python main.py`
5. Generated audio files will be saved in the current directory.

## Evaluation

You can evaluate the performance of your model by comparing the generated audio with the original audio recordings. You can use metrics such as the mean squared error (MSE) or the structural similarity index (SSIM) to quantify the similarity between the generated and original audio.

## Fine-tuning

Once you have a working model, you can fine-tune it by training it on a larger dataset, or by using more advanced techniques such as adversarial training or Wasserstein GANs. 

## Note
It's important to note that GANs are complex models that require a lot of data and computational resources to train effectively. Also, voice transfer is still in the early stages of research, so it's not guaranteed that you will get the desired results.
