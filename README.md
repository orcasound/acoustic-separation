"# Gsoc_orcasound" 
Currently a working GUI to process audio files is working. 
Please ignore calculator.py as it is a template I created to use on the GUI.
A sample raw audio called "20181202_25sec.mp3" is provided as input example.
"trial.m" filters "20181202_25sec.mp3", created power density frequency plots for each filter saved in the folder Plots,
and created filtered wav files saved in MATLAB_filtered_audios folder.

The GUI is in "processing.py". It calls "transforms.py" where all the processing techniques are computed.
"screencapture.mp4" is a demo of how GUI works.
"Orca_background.jpeg" is the background image of the GUI

1. One can use the [fine-tuned 2stems spleeter model](https://drive.google.com/drive/folders/1WU3_z-U9m1UX2jREFHwOH99pMKhPEdnu?usp=sharing) to separate orca vocalizations from the audio. Make sure that the folder named "pretrained_models" is present in the same directory containing the "processing.py" file.
2. One can use the [zeroshot model](https://drive.google.com/drive/folders/1RouwHsGsMs8n3l_jF8XifWtbPzur_YQS?usp=sharing). Create a folder "checkpoint" in the same directory containing the "processing.py" file and download the models in this folder.
