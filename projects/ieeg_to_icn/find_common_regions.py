"""
    This script is to identify the common brain regions found across all patients.

    By June 2022, the first 20 most popular regions were:
        G    18
        PST  10
        AST  10
        MST   9
        TT    8
        OF    8
        LPT   7
        LAT   6
        LF    5
        F     5
        PT    5
        SF    5
        AT    5
        LAF   4
        LAD   3
        ABT   3
        MBT   3
        MT    3
        LSF   3
        LFP   3
    This means that there are 18 patients that had electrodes in region G,
    10 in region PST, 10 in region AST, 9 in region MST, etc.
"""

import os
import pandas as pd
from bgreg.native.datapaths import datastore_path
from bgreg.utils.dataproc.bidsproc import get_regions_dict

# Get subjects' id from ecog-data patient_dir
subjects = []
for f in os.listdir(datastore_path):
    if f[:3] == 'sub':
        print("Checking {}".format(f))
        f_split = f.split('-')
        subjects.append(f_split[1])

# get the regions in each patient and store it
# in the dictionary common_regions
common_regions = {}
for sub in subjects:
    # FIXME: June 13, 2023
    #  problem is that the get_raw_signal_bids() method in utils.sigio
    #  requires to indicate the run to read from. Default run=1, and not all
    #  patient files have a run=1 --> Need to fix with automation
    common_regions[sub] = get_regions_dict(sub)  # elecs.electrode_locs(sub)['brainRegions']

# count the number of electrodes by region across all patients
elec_by_region_count = {}
# as we iterate through all the patients,
# we will also count the number of region occurrence
region_count = {}
for sub in subjects:
    for region, elec_range in common_regions[sub].items():
        elec_count = len(elec_range)
        elec_by_region_count.setdefault(region, 0)
        region_count.setdefault(region, 0)
        elec_by_region_count[region] += elec_count
        region_count[region] += 1

# number of electrodes per region across all patients
df_elec_to_region_count = pd.DataFrame.from_dict(elec_by_region_count, orient='index')
# count of regions across patients
df_region_count = pd.DataFrame.from_dict(region_count, orient='index')
# print(df_elec_to_region_count.sort_values(by=0, ascending=False).head(20))
print(df_region_count.sort_values(by=0, ascending=False).head(20))
