import glob
from pathlib import Path

import numpy as np
import pandas as pd
import librosa

from scipy.io import wavfile


class DataGenerator:
    """Helper class for generating 'separation' dataset."""

    def __init__(self, train_orca_calls_path, train_noise_path, val_orca_calls_path, val_noise_path):
        """Default constructor.
        :param train_orca_calls_path: Path to the directory containing isolated orca vocals to be used for training.
        :param train_orca_calls_path: Path to the directory containing noise (from boats,ships,etc.) to be overlapped
                                      with orca vocals for training.
        :param val_orca_calls_path: Path to the directory containing isolated orca vocals to be used for validation.
        :param val_orca_calls_path: Path to the directory containing noise (from boats,ships,etc.) to be overlapped
                                    with orca vocals for validation."""

        self.train_orca_calls_path = train_orca_calls_path
        self.train_noise_path = train_noise_path
        self.val_orca_calls_path = val_orca_calls_path
        self.val_noise_path = val_noise_path
        self.sample_rate = 44100

    def _mono_to_stereo(self, wave):
        """Function to convert mono (single channel) data to stereo (2 channel) data"""

        tone_y_stereo = np.vstack((wave, wave))
        tone_y_stereo = tone_y_stereo.transpose()
        return tone_y_stereo

    def _load_audio(self):
        """Loads all the audio files present in the training and the validation directories."""

        self.orca_files_train = glob.glob(self.train_orca_calls_path + "/*.wav") + glob.glob(self.train_orca_calls_path + "/*.mp3")
        self.noise_files_train = glob.glob(self.train_noise_path + "/*.wav") + glob.glob(self.train_noise_path + "/*.mp3")
        self.orca_files_val = glob.glob(self.val_orca_calls_path + "/*.wav") + glob.glob(self.val_orca_calls_path + "/*.mp3")
        self.noise_files_val = glob.glob(self.val_noise_path + "/*.wav") + glob.glob(self.val_noise_path + "/*.mp3")
        print(self.orca_files_train)

        self.training_callaudio = {}
        self.training_noiseaudio = {}
        self.val_callaudio = {}
        self.val_noiseaudio = {}

        for i, file in enumerate(self.orca_files_train, 1):
            call, _ = librosa.load(file, sr=self.sample_rate)
            self.training_callaudio["call" + str(i)] = call

        for i, file in enumerate(self.noise_files_train, 1):
            noise, _ = librosa.load(file, sr=self.sample_rate)
            self.training_noiseaudio["noise" + str(i)] = noise

        for i, file in enumerate(self.orca_files_val, 1):
            call, _ = librosa.load(file, sr=self.sample_rate)
            self.val_callaudio["call" + str(i)] = call

        for i, file in enumerate(self.noise_files_val, 1):
            noise, _ = librosa.load(file, sr=self.sample_rate)
            self.val_noiseaudio["noise" + str(i)] = noise

    def _generate(self, dataset_size, output_path, train, channels):
        """Generates the dataset and saves the audio files in the specified directory.
        :param dataset_size: Size of the dataset (can be training or validation dataset).
        :param output_path: Directory where the dataset will be saved.
        :param train: A boolean value. Set True while generating with training dataset and False while generating
                      validation dataset.
        :param channels: Number of output channels- single channel (mono) or double channel (stereo)."""

        noiseaudio = self.training_noiseaudio if train else self.val_noiseaudio
        callaudio = self.training_callaudio if train else self.val_callaudio
        noise_files = self.noise_files_train if train else self.noise_files_val
        orca_files = self.orca_files_train if train else self.orca_files_val
        directory = "Train" if train else "Validation"
        complete_path = output_path+"/"+directory

        dataset = pd.DataFrame(columns=['mix_path', 'orca_path', 'other_path', 'duration'])

        noise_num = 1
        call_num = 1

        for i in range(dataset_size):
            path = complete_path + "/audio_" + str(i + 1)
            Path(path).mkdir(parents=True, exist_ok=True)

            noise = noiseaudio["noise" + str(noise_num)]
            orca = callaudio["call" + str(call_num)]

            max_dur = min(int(noise.shape[0] / self.sample_rate), int(orca.shape[0] / self.sample_rate))
            if 5 < max_dur < 10:
                min_dur = 5
            else:
                if (max_dur // 10)*10 < max_dur:
                    min_dur = (max_dur // 10)*10
                else:
                    min_dur = max_dur-5

            duration = np.random.randint(min_dur, max_dur)

            start_time_orca = np.random.randint(0, int(orca.shape[0] / self.sample_rate) - duration)
            start_time_noise = np.random.randint(0, int(noise.shape[0] / self.sample_rate) - duration)

            orca_amplitude = np.random.uniform(0.5, 1.0)
            noise_amplitude = np.random.uniform(0.5, 1.0)

            orca = orca_amplitude * orca[(start_time_orca) * self.sample_rate: (start_time_orca + duration) * self.sample_rate]
            noise = noise_amplitude * noise[
                                      (start_time_noise) * self.sample_rate: (start_time_noise + duration) * self.sample_rate]

            mixed = orca + noise

            if channels == 2:
                orca = self._mono_to_stereo(orca)
                noise = self._mono_to_stereo(noise)
                mixed = self._mono_to_stereo(mixed)

            wavfile.write(complete_path + '/audio_' + str(i + 1) + '/orca.wav', self.sample_rate, orca)
            wavfile.write(complete_path + '/audio_' + str(i + 1) + '/noise.wav', self.sample_rate, noise)
            wavfile.write(complete_path + '/audio_' + str(i + 1) + '/mixed.wav', self.sample_rate, mixed)

            df = {'mix_path': directory + '/audio_' + str(i + 1) + '/mixed.wav',
                  'orca_path': directory + '/audio_' + str(i + 1) + '/orca.wav',
                  'other_path': directory + '/audio_' + str(i + 1) + '/noise.wav', 'duration': float(duration)}

            dataset = dataset.append(df, ignore_index=True)

            print(noise_num, call_num)

            if noise_num == len(noise_files) and call_num == len(orca_files):
                noise_num = 1
                call_num = 1

            else:
                if call_num == len(orca_files):
                    call_num = 1
                    noise_num += 1

                else:
                    call_num += 1

        dataset.to_csv(output_path+'/Orca_'+directory+'.csv', index=False)

    def generate_dataset(self, training_dataset_size, validation_dataset_size, output_path, channels=2):
        """Generates the dataset and saves the audio files in the specified directory.
        :param training_dataset_size: Size of the training dataset.
        :param validation_dataset_size: Size of the validation dataset.
        :param output_path: Directory where the dataset will be saved.
        :param channels: Number of output channels- single channel (mono) or double channel (stereo)."""

        self._load_audio()
        self._generate(training_dataset_size, output_path, train=True, channels=channels)
        self._generate(validation_dataset_size, output_path, train=False, channels=channels)

