"""
    This script contains methods to handle the patient_dir tree of different projects,
    which can be customized in native.datapaths. Currently only supporting the projects
    ieeg_to_icn and gnn_to_soz

    It also contains the methods to save and retrieve the files
    for the coherence and intrinsic coherence matrices.
"""
from bgreg.native.datapaths import *
import numpy as np


def ieeg_to_icn():
    """
    Initialize patient_dir tree for data management.
    :return:
    """
    check_and_create_dir(datastore_path)
    check_and_create_dir(datarecord_path)


def coh_icn(subject):
    """
    Initialize subdirectory structure for each patient
    to store coherence and ICN matrices in their respective directories.

    :param subject: (string) ID of patient
    :return:
    """
    # Create directories for coherence matrix data storage (see README.md)
    coh_mat_dirs = [coh_mat_dir_preictal, coh_mat_dir_preictal_images,
                    coh_mat_dir_ictal, coh_mat_dir_ictal_images,
                    coh_mat_dir_postictal, coh_mat_dir_postictal_images]
    for _coh_mat_dir in coh_mat_dirs:
        check_and_create_dir(os.path.join(_coh_mat_dir, f'{subject}'))

    # Create directories for intrinsic coherence network data storage (see README.md)
    icn_dirs = [icn_dir_preictal, icn_dir_preictal_images_mat,
                icn_dir_preictal_images_icn,
                icn_dir_ictal, icn_dir_ictal_images_mat,
                icn_dir_ictal_images_icn,
                icn_dir_postictal, icn_dir_postictal_images_mat,
                icn_dir_postictal_images_icn]
    for _icn_dir in icn_dirs:
        check_and_create_dir(os.path.join(_icn_dir, f'{subject}'))


def check_and_create_dir(dir_path):
    """
    Create patient_dir if it doesn't exist
    :param dir_path: (str) patient_dir path
    :return:
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def save_matrices(filename, matrices, mat_type='coh', trace_type='preictal'):
    """
    Save matrices-trace object into files.
    :param filename: (str) filename including file format (i.e., .npy)
    :param matrices: (dict) dictionary with coherence matrix values
    :param mat_type: (str) matrix type, supported values: coh and icn
    :param trace_type: (str) signal trace type, supported values: preictal, ictal and postictal
    :return:
    """
    if mat_type == 'coh' and trace_type == 'preictal':
        save_matrices_to_file(filename, matrices, coh_mat_dir_preictal)
    elif mat_type == 'coh' and trace_type == 'ictal':
        save_matrices_to_file(filename, matrices, coh_mat_dir_ictal)
    elif mat_type == 'coh' and trace_type == 'postictal':
        save_matrices_to_file(filename, matrices, coh_mat_dir_postictal)

    elif mat_type == 'icn' and trace_type == 'preictal':
        save_matrices_to_file(filename, matrices, icn_dir_preictal)
    elif mat_type == 'icn' and trace_type == 'ictal':
        save_matrices_to_file(filename, matrices, icn_dir_ictal)
    elif mat_type == 'icn' and trace_type == 'postictal':
        save_matrices_to_file(filename, matrices, icn_dir_postictal)
    else:
        print("utils.dataproc.dirtree.save_matrices() - error saving matrix")


def save_matrices_to_file(filename, matrices, directory):
    """
    Save matrices to files with NumPy
    :param filename: (str) file name including extension (i.e., .npy)
    :param matrices: (mne.Raw object) set of matrices
    :param directory: (native.datapaths object) patient_dir where to save the file
    :return:
    """
    np.save(os.path.join(directory, filename), matrices)


def save_matrix_to_image_file(subject, fig, title, mat_type, trace_type):
    """
    Save matrices into svg files
    :param subject: (str) patient id
    :param fig: (matplotlib.fig object)
    :param title: (str) title for figure and file
    :param mat_type: (str) type of matrix, supported values: 'coh', 'icn'
    :param trace_type: (str) signal trace type, supported values: 'preictal', 'ictal', 'postictal'
    :return:
    """
    if mat_type == 'coh' and trace_type == 'preictal':
        dirpath = os.path.join(coh_mat_dir_preictal_images, subject)
    elif mat_type == 'coh' and trace_type == 'ictal':
        dirpath = os.path.join(coh_mat_dir_ictal_images, subject)
    elif mat_type == 'coh' and trace_type == 'postictal':
        dirpath = os.path.join(coh_mat_dir_postictal_images, subject)

    elif mat_type == 'icn' and trace_type == 'preictal':
        dirpath = os.path.join(icn_dir_preictal_images_mat, subject)
    elif mat_type == 'icn' and trace_type == 'ictal':
        dirpath = os.path.join(icn_dir_ictal_images_mat, subject)
    elif mat_type == 'icn' and trace_type == 'postictal':
        dirpath = os.path.join(icn_dir_postictal_images_mat, subject)

    else:
        dirpath = '.'
    figtitle = os.path.join(dirpath, title + '.svg')
    fig.savefig(figtitle, format='svg', pad_inches=0, bbox_inches='tight', dpi=300)
    print("Figure {} saved successfully".format(title))


def save_icn_to_image_file(subject, fig, title, trace_type):
    if trace_type == 'preictal':
        dirpath = os.path.join(icn_dir_preictal_images_icn, subject)
    elif trace_type == 'ictal':
        dirpath = os.path.join(icn_dir_ictal_images_icn, subject)
    elif trace_type == 'postictal':
        dirpath = os.path.join(icn_dir_postictal_images_icn, subject)
    else:
        dirpath = '.'
    figtitle = os.path.join(dirpath, title + '.svg')
    fig.savefig(figtitle, format='svg', pad_inches=0, bbox_inches='tight', dpi=300)
    print("Figure {} saved successfully".format(title))


def get_matrices_file(filename, mat_type='coh', trace_type='preictal'):
    if mat_type == 'coh' and trace_type == 'preictal':
        return os.path.join(coh_mat_dir_preictal, filename)
    elif mat_type == 'coh' and trace_type == 'ictal':
        return os.path.join(coh_mat_dir_ictal, filename)
    elif mat_type == 'coh' and trace_type == 'postictal':
        return os.path.join(coh_mat_dir_postictal, filename)

    elif mat_type == 'icn' and trace_type == 'preictal':
        return os.path.join(icn_dir_preictal, filename)
    elif mat_type == 'icn' and trace_type == 'ictal':
        return os.path.join(icn_dir_ictal, filename)
    elif mat_type == 'icn' and trace_type == 'postictal':
        return os.path.join(icn_dir_postictal, filename)
    else:
        print("utils.dataproc.dirtree.get_matrices_file() - error retrieving matrices file")
