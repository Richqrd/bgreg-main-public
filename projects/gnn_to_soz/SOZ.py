import pandas as pd
import re
from pathlib import Path
from bgreg.native.datapaths import datastore_path


# get SOZ for the patients used in testing
def get_SOZ():
    # 25 patients only, others cannot be used due to no onset/offset time
    patients = ["jh101", "jh103", "jh108",
                "pt1", "pt2", "pt3", "pt6", "pt7", "pt8", "pt10", "pt12", "pt13", "pt14", "pt16",
                "umf001",
                "ummc_001", "ummc002", "ummc003", "ummc004", "ummc005", "ummc006", "ummc_007", "ummc009"]

    # some are named differently in excel vs in folder
    renamed = {"pt1": "pt01", "ummc_001": "ummc001", "ummc_007": "ummc007"}

    # get excel data
    data = pd.read_excel(Path(datastore_path, "sourcedata", "clinical_data_summary.xlsx"))

    # parse target soz
    SOZ = {}
    for patient in patients:
        if patient in renamed:
            name = renamed[patient]
        else:
            name = patient
        patient_data = data[data["dataset_id"] == patient]
        # use regex to split
        patient_soz = list(filter(None, re.split(";| |:|,|\n", patient_data["soz_contacts"].item().upper())))
        # convert 1-3 to 1, 2, 3
        patient_soz_single = []
        for soz in patient_soz:
            if '-' not in soz:
                patient_soz_single.append(soz)
            else:
                number = re.findall(r'\d+', soz)
                channel = re.findall(r'[A-Z]+', soz)
                for n in range(int(number[0]), int(number[1]) + 1):
                    patient_soz_single.append(channel[0] + str(n))
        SOZ[name] = patient_soz_single

    return SOZ
