import numpy as np
from sklearn.decomposition import PCA, FastICA


def pca_calculation(coh_matrices, nPCs):
    """
    Single PCA calculation of coherence matrices

    - coh_matrix: time series of analysis matrix (eg nElectrodes*nElecrodes*tTimepoints)
    - nPCs: number of PCs to calculate
    """

    coh_matrices_proc = np.zeros_like(coh_matrices)

    # Generate a single matrix by computing the mean
    # coherence across N coherence matrices in coh_matrix
    mean_coh_mat = np.mean(coh_matrices, axis=-1)

    # subtract the mean_coh_mat to all N coherence matrices
    for coh_matrix in np.arange(coh_matrices.shape[-1]):
        coh_matrices_proc[:, :, coh_matrix] = coh_matrices[:, :, coh_matrix] - mean_coh_mat

    # Create PCA model with nPCs
    pca = PCA(n_components=nPCs, svd_solver='full')

    X = np.reshape(coh_matrices_proc,
                   (coh_matrices_proc.shape[0] * coh_matrices_proc.shape[1],
                    coh_matrices_proc.shape[2]))
    tmp = pca.fit_transform(X)
    # Fit the PCA model to obtain matrices described by the nPCs
    pc = np.reshape(tmp,
                    (coh_matrices_proc.shape[0],
                     coh_matrices_proc.shape[1],
                     nPCs))

    # Explained variance represents the information explained using
    # a particular principal components (eigenvectors). Explained
    # variance is calculated as ratio of eigenvalue of a particular
    # principal component (eigenvector) with total eigenvalues
    pcEigV = pca.explained_variance_ratio_

    return pc, pcEigV


def ica_calculation(dataMatrixIn, nICs):
    """
    Single ICA calculation of coherence matrices
    - dataMatrixIn: time series of analysis matrix (eg nElectrodes*nElecrodes*tTimepoints)
    - nICs: number of ICs to calculate
    """
    # ICA, the random_state=0 param guarantees consistent IC computation over multiple runs
    ica = FastICA(n_components=nICs, random_state=0, max_iter=2000, tol=0.001)
    X = np.reshape(dataMatrixIn, (dataMatrixIn.shape[0] * dataMatrixIn.shape[1], dataMatrixIn.shape[2]))
    # Reconstruct and return signals (ICs)
    return np.reshape(ica.fit_transform(X), (dataMatrixIn.shape[0], dataMatrixIn.shape[1], nICs))


def project_onto_component(coh_matrices, icn_matrices, symmMat=True):
    """
    Project data (eg coherence matrices) onto components (PCs or ICs)

    - dataMatrix: nElectrodes*nElectrodes*tTimepoints
    - compMarix: nElectrodes*nElectrodes*nComponents
    - norm: True or False. Normalize projection if True

    Returns
    - projMatrix: tTimepoints*nComponents; projection onto each component
    """

    # Determine number of components and number of time points
    nRegions = np.shape(icn_matrices)[0]
    nCpts = np.shape(icn_matrices)[-1]
    tPts = coh_matrices.shape[-1]

    # If matrix is symmetrical take only half data and comp matrices so as not to double count off diagonal regions
    if symmMat:
        coh_matrices_proc = np.zeros_like(coh_matrices)
        icn_matrices_proc = np.zeros_like(icn_matrices)

        for ii in range(coh_matrices.shape[0]):
            for jj in range(ii, coh_matrices.shape[0]):
                coh_matrices_proc[ii, jj, :] = coh_matrices[ii, jj, :]
                icn_matrices_proc[ii, jj, :] = icn_matrices[ii, jj, :]
    else:
        coh_matrices_proc = coh_matrices
        icn_matrices_proc = icn_matrices

    dataLinear = np.reshape(coh_matrices_proc, (nRegions * nRegions, tPts))
    compLinear = np.reshape(icn_matrices_proc, (nRegions * nRegions, nCpts))

    projMatrix = np.dot(compLinear.T, dataLinear)

    return projMatrix


def find_ICs_from_PCA(coh_matrices, nPCs=10):
    """
    Find independent components based on number of significant PCs using marcenko-pastur limit
    First, we apply PCA to do dimensionality reduction to nPCs,
        and compute the eigenvalues from the resulted matrix.
    Then, we compute the eigenvalue threshold (marcenko-pasteur),
        and, use this threshold to discriminate eigenvalues computed in the first step.
        This will identify the number of most significant eigenvalues (and thus the vectors)
        from the matrix.
    Last, we apply ICA to the dimension-reduced matrix (PCs), which will result in the
        ICN network that we are looking for.

    :param coh_matrices: (ndarray) coherence matrix
    :param nPCs: (int) number of principal components
    :return:
    """
    # Apply dimensionality reduction and compute the
    # eigenvalue for each coherence matrix
    PCs, eigVs = pca_calculation(coh_matrices, nPCs=nPCs)

    # Get the eigenvalue threshold given by the Marcenko-Pasteur limit
    eigVthresh = eigenval_thresh(coh_matrices)

    # Identify the number of coherence matrices with
    # eigenvalues above the eigenvalue threshold
    nSig = len(np.where(eigVs >= eigVthresh)[0])

    # Apply ICA to the first nSig principal components
    return ica_calculation(PCs[:, :, :nSig], nICs=nSig)


def eigenval_thresh(coh_matrix):
    """
    Eigenvalue threshold, the number above which can be used to estimate the
    number of statistically significant components in the data matrix.
    """

    sz = coh_matrix.shape
    coh_matrix_2d = coh_matrix.reshape((sz[0] * sz[1], sz[2]))

    nRows = coh_matrix_2d.shape[0]
    nCols = coh_matrix_2d.shape[1]

    sigma2 = np.mean(np.var(coh_matrix_2d, axis=1))

    lam_max, lam_min = marcenko_pastur(nRows, nCols, sigma2)

    return lam_max


def marcenko_pastur(nRows, nCols, sigma2):
    """
    The Marcenko-Pasteur limit gives the distribution of eigenvalues expected by chance.

    Inputs:
        - nCols: number of columns in M
        - nRows: number of rows in M
        - sigma2: variance of elements in M
    """
    q = float(nRows) / float(nCols)

    lambda_max = sigma2 * (1 + np.sqrt(1 / q)) ** 2
    lambda_min = sigma2 * (1 - np.sqrt(1 / q)) ** 2

    return lambda_max, lambda_min


def get_icn(coh_matrices):
    sz = coh_matrices.shape
    # nSamples is the number of electrodes * electrodes
    nSamples = sz[0] * sz[1]
    # nFeatures is the number of coherence matrix segments
    nFeatures = sz[2]
    nPCs = min(nSamples, nFeatures) - 1
    icnMat = find_ICs_from_PCA(coh_matrices, nPCs)
    # FIXME: What is the icnProj used for?
    icnProj = project_onto_component(coh_matrices, icnMat, symmMat=True)

    return icnMat, icnProj
