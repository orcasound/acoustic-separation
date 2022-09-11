#!/usr/bin/env python
# coding: utf8



import atexit

from multiprocessing import Pool
from typing import Container, NoReturn

import numpy as np
import tensorflow as tf

from librosa.core import stft, istft
from scipy.signal.windows import hann

import json
from spleeter.audio.convertor import to_stereo
from spleeter.model import EstimatorSpecBuilder, InputProviderFactory, model_fn


SUPPORTED_BACKEND: Container[str] = ('auto', 'tensorflow', 'librosa')
""" """

def create_estimator(model_dir, params, MWF):
    """
        Initialize tensorflow estimator that will perform separation

        Params:
        - params: a dictionary of parameters for building the model

        Returns:
            a tensorflow estimator
    """
    # Load model.
    params['model_dir'] = model_dir
    params['MWF'] = MWF
    # Setup config
    session_config = tf.compat.v1.ConfigProto()
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config = tf.estimator.RunConfig(session_config=session_config)
    # Setup estimator
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=params['model_dir'],
        params=params,
        config=config
    )
    return estimator

class DataGenerator():
    """
        Generator object that store a sample and generate it once while called.
        Used to feed a tensorflow estimator without knowing the whole data at
        build time.
    """

    def __init__(self):
        """ Default constructor. """
        self._current_data = None

    def update_data(self, data):
        """ Replace internal data. """
        self._current_data = data

    def __call__(self):
        """ Generation process. """
        buffer = self._current_data
        while buffer:
            yield buffer
            buffer = self._current_data


def get_backend(backend: str) -> str:
    """
    """
    if backend not in SUPPORTED_BACKEND:
        raise ValueError(f'Unsupported backend {backend}')
    if backend == 'auto':
        if len(tf.config.list_physical_devices('GPU')):
            return 'tensorflow'
        return 'librosa'
    return backend


class Separator(object):
    """ A wrapper class for performing separation. """

    def __init__(
            self,
            params_descriptor,
            model_path,
            MWF: bool = False,
            stft_backend: str = 'auto',
            multiprocess: bool = False):
        """ Default constructor.

        :param params_descriptor: Descriptor for TF params to be used.
        :param MWF: (Optional) True if MWF should be used, False otherwise.
        """
        with open(params_descriptor, 'r') as stream:
            self._params = json.load(stream)
        self._model_path = model_path
        self._sample_rate = self._params['sample_rate']
        self._MWF = MWF
        self._tf_graph = tf.Graph()
        self._prediction_generator = None
        self._input_provider = None
        self._builder = None
        self._features = None
        self._session = None
        if multiprocess:
            self._pool = Pool()
            atexit.register(self._pool.close)
        else:
            self._pool = None
        self._tasks = []
        self._params['stft_backend'] = get_backend(stft_backend)
        self._data_generator = DataGenerator()

    def __del__(self):
        """ """
        if self._session:
            self._session.close()

    def _get_prediction_generator(self):
        """ Lazy loading access method for internal prediction generator
        returned by the predict method of a tensorflow estimator.

        :returns: generator of prediction.
        """
        if self._prediction_generator is None:
            estimator = create_estimator(self._model_path, self._params, self._MWF)

            def get_dataset():
                return tf.data.Dataset.from_generator(
                    self._data_generator,
                    output_types={
                        'waveform': tf.float32,
                        'audio_id': tf.string},
                    output_shapes={
                        'waveform': (None, 2),
                        'audio_id': ()})

            self._prediction_generator = estimator.predict(
                get_dataset,
                yield_single_examples=False)
        return self._prediction_generator

    def join(self, timeout: int = 200) -> NoReturn:
        """ Wait for all pending tasks to be finished.

        :param timeout: (Optional) task waiting timeout.
        """
        while len(self._tasks) > 0:
            task = self._tasks.pop()
            task.get()
            task.wait(timeout=timeout)

    def _separate_tensorflow(self, waveform: np.ndarray, audio_descriptor):
        """ Performs source separation over the given waveform with tensorflow
        backend.

        :param waveform: Waveform to apply separation on.
        :returns: Separated waveforms.
        """
        if not waveform.shape[-1] == 2:
            waveform = to_stereo(waveform)
        prediction_generator = self._get_prediction_generator()
        # NOTE: update data in generator before performing separation.
        self._data_generator.update_data({
            'waveform': waveform,
            'audio_id': np.array(audio_descriptor)})
        # NOTE: perform separation.
        prediction = next(prediction_generator)
        prediction.pop('audio_id')
        return prediction

    def _stft(self, data, inverse: bool = False, length=None):
        """ Single entrypoint for both stft and istft. This computes stft and
        istft with librosa on stereo data. The two channels are processed
        separately and are concatenated together in the result. The expected
        input formats are: (n_samples, 2) for stft and (T, F, 2) for istft.

        :param data:    np.array with either the waveform or the complex
                        spectrogram depending on the parameter inverse
        :param inverse: should a stft or an istft be computed.
        :returns:   Stereo data as numpy array for the transform.
                    The channels are stored in the last dimension.
        """
        assert not (inverse and length is None)
        data = np.asfortranarray(data)
        N = self._params['frame_length']
        H = self._params['frame_step']
        win = hann(N, sym=False)
        fstft = istft if inverse else stft
        win_len_arg = {
            'win_length': None,
            'length': None} if inverse else {'n_fft': N}
        n_channels = data.shape[-1]
        out = []
        for c in range(n_channels):
            d = np.concatenate(
                (np.zeros((N, )), data[:, c], np.zeros((N, )))
                ) if not inverse else data[:, :, c].T
            s = fstft(d, hop_length=H, window=win, center=False, **win_len_arg)
            if inverse:
                s = s[N:N+length]
            s = np.expand_dims(s.T, 2-inverse)
            out.append(s)
        if len(out) == 1:
            return out[0]
        return np.concatenate(out, axis=2-inverse)

    def _get_input_provider(self):
        if self._input_provider is None:
            self._input_provider = InputProviderFactory.get(self._params)
        return self._input_provider

    def _get_features(self):
        if self._features is None:
            provider = self._get_input_provider()
            self._features = provider.get_input_dict_placeholders()
        return self._features

    def _get_builder(self):
        if self._builder is None:
            self._builder = EstimatorSpecBuilder(
                self._get_features(),
                self._params)
        return self._builder

    def _get_session(self):
        if self._session is None:
            saver = tf.compat.v1.train.Saver()
            latest_checkpoint = tf.train.latest_checkpoint(self._model_path)
            self._session = tf.compat.v1.Session()
            saver.restore(self._session, latest_checkpoint)
        return self._session

    def _separate_librosa(self, waveform: np.ndarray, audio_id):
        """ Performs separation with librosa backend for STFT.
        """
        with self._tf_graph.as_default():
            out = {}
            features = self._get_features()
            # TODO: fix the logic, build sometimes return,
            #       sometimes set attribute.
            outputs = self._get_builder().outputs
            stft = self._stft(waveform)
            if stft.shape[-1] == 1:
                stft = np.concatenate([stft, stft], axis=-1)
            elif stft.shape[-1] > 2:
                stft = stft[:, :2]
            sess = self._get_session()
            outputs = sess.run(
                outputs,
                feed_dict=self._get_input_provider().get_feed_dict(
                    features,
                    stft,
                    audio_id))
            for inst in self._get_builder().instruments:
                out[inst] = self._stft(
                    outputs[inst],
                    inverse=True,
                    length=waveform.shape[0])
            return out

    def separate(self, waveform: np.ndarray, audio_descriptor=''):
        """ Performs separation on a waveform.

        :param waveform:            Waveform to be separated (as a numpy array)
        :param audio_descriptor:    (Optional) string describing the waveform
                                    (e.g. filename).
        """
        if self._params['stft_backend'] == 'tensorflow':
            print("Using tensorflow backend.")
            return self._separate_tensorflow(waveform, audio_descriptor)
        else:
            return self._separate_librosa(waveform, audio_descriptor)

    def return_source_dictionary(self, waveform: np.ndarray):
        """
        :param waveform: Waveform to be separated (as a numpy array)
        :return:
        """
        waveform = waveform.reshape((waveform.shape[0], 1)).astype(np.float32)
        sources = self.separate(waveform)
        return sources
