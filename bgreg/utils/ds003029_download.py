"""
    FIXME: this does not seem to be working correctly!
    Script to download the ds003029 dataset.

    Adam Li and Sara Inati and Kareem Zaghloul and Nathan Crone and
    William Anderson and Emily Johnson and Iahn Cajigas and Damian
    Brusko and Jonathan Jagid and Angel Claudio and Andres Kanner and
    Jennifer Hopp and Stephanie Chen and Jennifer Haagensen and
    Sridevi Sarma (2021). Epilepsy-iEEG-Multicenter-Dataset.
    OpenNeuro. [Dataset] doi: 10.18112/openneuro.ds003029.v1.0.3
"""
import openneuro
from bgreg.native.datapaths import os, datastore_path


def download_all():
    """
    Download all the files in dataset:
        Files: 679 Size: 10.32GB
    Storing patient_dir given in utils.native.datapaths.datastore_path
    :return:
    """
    openneuro.download(dataset="ds003029", target_dir=datastore_path)


def download_subject(subject):
    """
    Download individual patient files.
    Storing patient_dir given in utils.native.datapaths.datastore_path
    :param subject: str, patient id
    :return:
    """
    openneuro.download(dataset="ds003029", target_dir=datastore_path,
                       include=[f'sub-{subject}'])
