# Dataset Generator-
This directory contains a Python module “orca_dataset_generator.py” with a class named “DataGenerator” to generate a ‘separation’ dataset that can be used to train the Spleeter model. There is also a Python file, “generate_dataset.py” that one can explore to find out how to use this module and generate their own custom dataset.

The generated dataset is saved inside two directories at the location passed as an argument in the above function- Train and Validation. The audio files- mixed.wav (containing mixture of noise and orca vocals), orca.wav and noise.wav, for each example are stored in multiple sub-directories under ‘Train’ and ‘Validation.’

Additionally, this method also saves two csv files- ‘Orca_Train.csv’ and ‘Orca_Validation.csv’ containing description of the audio files. These csv files have four columns- ‘mix_path’ containing path to the mixture audio files, ‘orca_path’ containing path to the orca vocals, ‘other_path’ containing path to the noise audio files, and ‘duration’ containing duration of each of these audio files.

```
├── generate_dataset.py
├── orca_dataset_generator.py
|
├── training_data/
│  ├── calls/
│  │  ├── calls1.wav
│  │  ├── calls2.wav
│  │  ├── ...
│  └── noise/
│     ├── noise1.wav
│     ├── noise2.wav
│     └── ...
|
├── validation_data/
│  ├── calls/
│  │  ├── calls1.wav
│  │  ├── calls2.wav
│  │  ├── ...
│  └── noise/
│     ├── noise1.wav
│     ├── noise2.wav
│     └── ...
│
└── generated_dataset/
   ├── Train/
   │  ├── audio_1/
   │  │  ├── mixed.wav
   │  │  ├── orca.wav
   │  │  ├── noise.wav
   │  ├── audio_2/
   │  │  ├── mixed.wav
   │  │  ├── orca.wav
   │  │  ├── noise.wav
   │  └── ...
   ├── Validation/
   │  ├── audio_1/
   │  │  ├── mixed.wav
   │  │  ├── orca.wav
   │  │  ├── noise.wav
   │  ├── audio_2/
   │  │  ├── mixed.wav
   │  │  ├── orca.wav
   │  │  ├── noise.wav
   │  └── ...
   ├── Orca_Train.csv
   └── Orca_Validation.csv
```
