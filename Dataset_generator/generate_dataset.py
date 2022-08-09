from orca_dataset_generator import DataGenerator
if __name__ == "__main__":

    """One can create an instance of the “DataGenerator” class. It takes 4 arguments-
    1.	train_orca_calls_path- Path to the directory containing isolated orca vocals to be used for training.
    2.	train_noise_path- Path to the directory containing noise (from boats,ships,etc.) to be overlapped with
        orca vocals for training.
    3.	val_orca_calls_path- Path to the directory containing isolated orca vocals to be used for validation.
    4.	val_noise_path- Path to the directory containing noise (from boats,ships,etc.) to be overlapped with
        orca vocals for validation.
    """

    generator = DataGenerator(train_orca_calls_path="training_data/calls",
                              train_noise_path="training_data/noise",
                              val_orca_calls_path="validation_data/calls",
                              val_noise_path="validation_data/noise")

    """Next one can call the member function “generate_dataset” to generate their custom dataset. This function also
        takes 4 parameters as arguments-
    1.	training_dataset_size- Size of the training dataset.
    2.	validation_dataset_size- Size of the validation dataset.
    3.	output_path- Directory where you want to save the dataset.
    4.	channels- The number of channels- single channel (mono) or double channel (stereo), you want
        in the generated audio files.
    """
    generator.generate_dataset(training_dataset_size=500,
                               validation_dataset_size=100,
                               output_path="generated_dataset",
                               channels=2)
