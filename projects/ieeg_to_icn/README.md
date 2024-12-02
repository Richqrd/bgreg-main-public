# iEEG to ICN

This project is summarized in this figure

![](../../../docs/source/_figures/poster.png)

(_intracranial electroencephalography to intrinsic coherence networks_)

This project provides a framework for creating cross-regional brain subnetwork (CRBS) representations in the form of intrinsic coherence networks (ICN [1]) from intracranial electroncephalography (iEEG) signals data.

It operates as a pipeline for:
- reading BIDS-iEEG data files [2].
- processing iEEG signals (i.e., segmentation, filtering).
- computing measurements of co-activity of the iEEG signals.
- computing CRBS representations: representing iEEG signals as graphs.
- visualizing iEEG signals, measurements of co-activity, and CRBS representations.

[1] ICN as defined in: _Kirkby, L.A., Luongo, F.J., Lee, M.B., Nahum, M., Van Vleet, T.M., Rao, V.R., Dawes, H.E., Chang, E.F. and Sohal, V.S., 2018. An amygdala-hippocampus subnetwork that encodes variation in human mood. Cell, 175(6), pp.1688-1700._

[2] _Holdgraf, C., Appelhoff, S., Bickel, S., Bouchard, K., D’Ambrosio, S., David, O., Devinsky, O., Dichter, B., Flinker, A., Foster, B.L. and Gorgolewski, K.J., 2019. iEEG-BIDS, extending the Brain Imaging Data Structure specification to human intracranial electrophysiology. Scientific data, 6(1), pp.1-6._

***
# Walkthrough

Here we showcase an example on 
- how to set up the directory tree structure for one project, 
- how to read a BIDS-iEEG patient record, how to use the functions in ```pipeline.py``` to read and process the data, compute the measurements of co-activity, compute the CRBS representations, and
- how to visualize the data . 

Any and all parts of the code are intended to serve as templates from where you can build upon.

### To get started

First, you must create the directory tree where _data_-related files will be stored. By default, all the data-related files will be stored under the ```~/ieeg_to_crbs/data``` directory. This directory is automatically created upon importing ```native/datapaths.py```. Check the ```native/README.md``` file fore more information on datapaths.


### The dataset
For this walkthrough, we use the data of the first run from pt6 in dataset from https://openneuro.org/datasets/ds003029/versions/1.0.3. You ought to download at least one patient dataset to continue with the execution of the pipeline. You can simply download from the link or use our API (```~/ieeg_to_crbs/projects/download_dataset.py```, or ```~/ieeg_to_crbs/utils/dataproc/ds003029_download.py```)

This is a dataset of human subjects with epilepsy that underwent epileptogenic-zone resective surgery. 
These are BIDS iEEG recordings that span 2-3min per record. 
The dataset contains information on the clinically identified epileptogenic zone for each patient.
Here we will create CRBS representations marking the epileptogenic-zone in the diagrams.

For more information about the dataset refer to [3]:

[3] _Adam Li and Sara Inati and Kareem Zaghloul and Nathan Crone and William Anderson and Emily Johnson and Iahn Cajigas and Damian Brusko and Jonathan Jagid and Angel Claudio and Andres Kanner and Jennifer Hopp and Stephanie Chen and Jennifer Haagensen and Sridevi Sarma (2021). Epilepsy-iEEG-Multicenter-Dataset. OpenNeuro. [Dataset] doi: 10.18112/openneuro.ds003029.v1.0.3_
### Running the pipeline

```main.py```  is a template for executing the pipeline for one patient.

```$ PYTHONPATH=. python3 projects/ieeg_to_icn/main.py```

Note: The epileptogenic (or seizure onset) zone of pt6, and from all the patients, can be found in the Excel sheet found in the repository ```data/sourcedata/clinical_data_summary.xlsx```.

Example of coherence matrix generated with the pipeline:

![](../../docs/source/_figures/cohmat_example.svg)

Example of ICN:

![](../../docs/source/_figures/icn_example.svg)

### What now?

The script ```ieeg_to_icn_pipeline.py``` contains all the functions for running the pipeline. Please refer to it to better understand how everything is connected and we are using the APIs from ```utils.dataproc```.
## Important note

All executable scripts, that is, scripts under the ```projects``` directory, are intended to be run from root directory position: the root folder of this repository (ieeg_to_crbs).

If you are using an IDE as PyCharm, you need to edit the ```Working directory``` configuration and set it to the root directory: ```~/ieeg_to_crbs/```.

If you are executing from a UNIX Terminal, you can set the ```PYTHONPATH``` to the root directory: ```$ PYTHONPATH=~/ieeg_to_crbs python3 projects/script.py```