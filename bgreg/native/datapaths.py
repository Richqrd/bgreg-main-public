import os
import json
import bgreg


def get_brain_greg_dir():
    """
    Return path to brain-greg directory
    """
    from sys import platform
    if platform == "win32":
        cwd_list = os.getcwd().split("\\")
        brain_greg_dir = ["\\"]
    else:
        cwd_list = os.getcwd().split("/")
        brain_greg_dir = ["/"]

    for item in cwd_list:
        brain_greg_dir.append(item)
        if item == "bgreg":
            return os.path.join(*brain_greg_dir)
    return os.path.join(*brain_greg_dir)


with open(os.path.join(os.path.dirname(bgreg.__file__), os.path.join("native", "variables.json")), "r") as f:
    variables = json.load(f)

datapath = os.path.join(get_brain_greg_dir(), variables["datastore_path"])

# Directory where data is (already) downloaded
# See README.md for instructions on how to download the data
datastore_path = os.path.join(datapath, variables["ds003029"])

# Directory for files created for later processing (see README.md)
datarecord_path = os.path.join(datapath, variables["ds003029_processed"])

# Directory to store coherence matrices
coh_mat_dir = os.path.join(datarecord_path, variables["coh_mat_dir"])
coh_mat_dir_whole = os.path.join(coh_mat_dir, variables["coh_mat_dir_whole"])
coh_mat_dir_whole_images = os.path.join(coh_mat_dir_whole, "images")
coh_mat_dir_preictal = os.path.join(coh_mat_dir, variables["coh_mat_dir_preictal"])
coh_mat_dir_preictal_images = os.path.join(coh_mat_dir_preictal, "images")
coh_mat_dir_ictal = os.path.join(coh_mat_dir, variables["coh_mat_dir_ictal"])
coh_mat_dir_ictal_images = os.path.join(coh_mat_dir_ictal, "images")
coh_mat_dir_postictal = os.path.join(coh_mat_dir, variables["coh_mat_dir_postictal"])
coh_mat_dir_postictal_images = os.path.join(coh_mat_dir_postictal, "images")

# Directory to store ICNs
# FIXME: should we get rid of the "whole" signal trace?
icn_dir = os.path.join(datarecord_path, variables["icn_dir"])
icn_dir_whole = os.path.join(icn_dir, variables["icn_dir_whole"])
icn_dir_whole_images_mat = os.path.join(icn_dir_whole, os.path.join("images", "mat"))
icn_dir_whole_images_icn = os.path.join(icn_dir_whole, os.path.join("images", "icn"))
icn_dir_preictal = os.path.join(icn_dir, variables["icn_dir_preictal"])
icn_dir_preictal_images_mat = os.path.join(icn_dir_preictal, os.path.join("images", "mat"))
icn_dir_preictal_images_icn = os.path.join(icn_dir_preictal, os.path.join("images", "icn"))
icn_dir_ictal = os.path.join(icn_dir, variables["icn_dir_ictal"])
icn_dir_ictal_images_mat = os.path.join(icn_dir_ictal, os.path.join("images", "mat"))
icn_dir_ictal_images_icn = os.path.join(icn_dir_ictal, os.path.join("images", "icn"))
icn_dir_postictal = os.path.join(icn_dir, variables["icn_dir_postictal"])
icn_dir_postictal_images_mat = os.path.join(icn_dir_postictal, os.path.join("images", "mat"))
icn_dir_postictal_images_icn = os.path.join(icn_dir_postictal, os.path.join("images", "icn"))

# Directory to store pickle signal traces (preictal, ictal, postictal splits)
preprocessed_data_dir = os.path.join(datarecord_path, variables["preprocessed_data"])
graph_representation_elements_dir = os.path.join(datarecord_path, variables["graph_representation_elements"])
