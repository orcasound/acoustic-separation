import os
import sys, getopt
import numpy as np
import librosa
import torch
from .utils import prepprocess_audio
from torch.utils.data import DataLoader
from .models.asp_model import ZeroShotASP, SeparatorModel, AutoTaggingWarpper
from . import htsat_config as htsat_config
from .models.htsat import HTSAT_Swin_Transformer
from .data_processor import OrcaDataset
from .sed_model import SEDWrapper
import pytorch_lightning as pl
from . import config as config


def inference(test_track, zeroshot_checkpoint, htsat_checkpoint, query_directory):
    # set exp settings
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda")
    test_track = test_track[:, None]

    # convert the track into 32000 Hz sample rate
    test_track = prepprocess_audio(
        test_track,
        44100, config.sample_rate,
        config.test_type
    )
    test_tracks = []
    temp = [test_track]
    for dickey in config.test_key:
        temp.append(test_track)
    temp = np.array(temp)
    test_tracks.append(temp)
    dataset = OrcaDataset(tracks=test_tracks)  # the action is similar to musdbdataset, reuse it
    loader = DataLoader(
        dataset=dataset,
        num_workers=1,
        batch_size=1,
        shuffle=False
    )
    # obtain the samples for query
    queries = []
    for query_file in os.listdir(query_directory):
        f_path = os.path.join(query_directory, query_file)
        if query_file.endswith(".wav"):
            temp_q, fs = librosa.load(f_path, sr=None)
            temp_q = temp_q[:, None]
            temp_q = prepprocess_audio(
                temp_q,
                fs, config.sample_rate,
                config.test_type
            )
            temp = [temp_q]
            for dickey in config.test_key:
                temp.append(temp_q)
            temp = np.array(temp)
            queries.append(temp)

    sed_model = HTSAT_Swin_Transformer(
        spec_size=htsat_config.htsat_spec_size,
        patch_size=htsat_config.htsat_patch_size,
        in_chans=1,
        num_classes=htsat_config.classes_num,
        window_size=htsat_config.htsat_window_size,
        config=htsat_config,
        depths=htsat_config.htsat_depth,
        embed_dim=htsat_config.htsat_dim,
        patch_stride=htsat_config.htsat_stride,
        num_heads=htsat_config.htsat_num_head
    )
    at_model = SEDWrapper(
        sed_model=sed_model,
        config=htsat_config,
        dataset=None
    )
    ckpt = torch.load(htsat_checkpoint, map_location="cpu")
    at_model.load_state_dict(ckpt["state_dict"])

    trainer = pl.Trainer(
        gpus=0
    )
    # accelerator = 'gpu', devices = 1

    # obtain the latent embedding as query
    avg_dataset = OrcaDataset(tracks=queries)
    avg_loader = DataLoader(
        dataset=avg_dataset,
        num_workers=1,
        batch_size=1,
        shuffle=False
    )
    at_wrapper = AutoTaggingWarpper(
        at_model=at_model,
        config=config,
        target_keys=config.test_key
    )
    trainer.test(at_wrapper, avg_loader)
    avg_at = at_wrapper.avg_at

    # import seapration model
    model = ZeroShotASP(
        channels=1, config=config,
        at_model=at_model,
        dataset=dataset
    )
    # resume checkpoint
    ckpt = torch.load(zeroshot_checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    exp_model = SeparatorModel(
        model=model,
        config=config,
        target_keys=config.test_key,
        avg_at=avg_at,
        using_wiener=False,
        calc_sdr=False,
        output_wav=True
    )
    trainer.test(exp_model, loader)
    output = exp_model.get_output()
    return output
