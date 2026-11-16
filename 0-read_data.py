import os
import hmp
from pathlib import Path
import pandas as pd
from mne.channels import make_standard_montage
import shutil

data_dir =  Path("../data")
participants_file = data_dir / "participants.tsv"

participants = pd.read_csv(participants_file, sep="\t")

# Identify participants without an events.tsv file
remove_subjects = []
for subject_dir in data_dir.glob("sub-*"):
    if not subject_dir.is_dir():
        continue
    subject = subject_dir.name
    if subject == "sub-105": #splitted in two recordings
        remove_subjects.append(subject)
        continue
    eeg_dir = subject_dir / "eeg"
    if not eeg_dir.exists() or not list(eeg_dir.glob("*events.tsv")):
        remove_subjects.append(subject)

# Delete all identified subject folders
for subject in remove_subjects:
    subject_dir = data_dir / subject

    if subject_dir.exists():
        shutil.rmtree(subject_dir)
        print(f"Deleted folder: {subject_dir}")

# Remove subjects from participants.tsv
participants = participants[
    ~participants["participant_id"].isin(remove_subjects)
]

# Save updated participants.tsv
participants.to_csv(participants_file, sep="\t", index=False)

print("\nRemoved subjects:", remove_subjects)
print("Total removed:", len(remove_subjects))
print("Remaining participants:", len(participants))
print("participants.tsv successfully updated.")

# Replace empty cells with n/a
for events_file in data_dir.rglob("*_events.tsv"):
    df = pd.read_csv(events_file, sep="\t", dtype=str)

    # Identify all empty or whitespace-only cells
    mask = df.isna() | df.apply(
        lambda col: col.str.strip().eq("")
    )

    if mask.any().any():
        n_replaced = mask.sum().sum()

        df = df.fillna("n/a")
        df = df.replace(r"^\s*$", "n/a", regex=True)

        df.to_csv(
            events_file,
            sep="\t",
            index=False,
            na_rep="n/a"
        )


# Adapt column names in events.tsv
# for file in data_dir.rglob("*.tsv"):
#     df = pd.read_csv(file, sep="\t")
#     if "trigger" not in df.columns:
#         continue
#     df.rename(columns={
#         "trial_type": "temp_column",
#         "event_type": "trial_type"
#     }, inplace=True)
#     df.rename(columns={
#         "temp_column": "trial_type",
#         "trigger": "value"
#     }, inplace=True)
#     df.to_csv(file, sep="\t", index=False)

for file in data_dir.rglob("*.tsv"):
    df = pd.read_csv(file, sep="\t")
    if "trial_type.1" not in df.columns:
        continue
    df.rename(columns={
        "trial_type": "event_type",
        "trial_type.1": "trial_type"
    }, inplace=True)
    df.to_csv(file, sep="\t", index=False)

# All elements in the dictionaries are expected to follow the format categorie/...
# For example for the event we want to center on we can describe them in the following way:
centering_id = {
    "stimulus/choice":253,
    "stimulus/display":252,
}

# Then we declare the triggers for which we want the time, here only the response trigger is important
event_id = {"response/left":234,
            "response/right":235}

# BIDS argument, root and selection (datatype, session, task, ...)
# Explore the docstrings of the hmp.io.read_bids_raw to see what selection you can perform
bids_kwargs = dict(
    root = os.path.join("..","data"), #Mandatory path to the data
    datatypes=['eeg'],#Mandatory, type of data we want to model, could be `meg`
    tasks=["categorisation"],# Here the task we want to analyze
)

# Preprocessing arguments, can have low_pass, high_pass, sfreq, reference and which channels to pick
# You can also use your own preprocessing function (see `section Applying your own preprocessing`)
# Explore the docstrings of the hmp.io.read_bids_raw to see what preprocessing steps are included
# Or how to define your own preprocessing function


# Epoching arguments, anything that can be used by mne.Epoch and that will define
# the epoching performed by HMP
epoching = dict(
    tmin=-.25,# Time in seconds to keep before a trigger of `centering_id` for baseline correction
    tmax=2.5,#Up to when in seconds, after a trigger of `centering_id`, should we look for a response_time end (e.g. response) event
    )


# Declaring your own preprocessing function,
# The two nex lines should always be the same
# Only the content of the fonction and the preprocessing_kwargs can be changed
def mypreprocessing(data, montage, events, verbose,#Mandatory arguments, passed internally
                    preprocessing_kwargs#preprocessing_kwargs should contain the parameters you need
                   ):
    data.load_data()
    # Set the reference
    data = data.set_eeg_reference(preprocessing_kwargs["reference"])
    # Filter the data
    data = data.filter(h_freq = preprocessing_kwargs["lowpass"],
                       l_freq=preprocessing_kwargs["highpass"])
    data = data.pick_channels(preprocessing_kwargs["pick_channels"])
    # Set montage manually
    montage = make_standard_montage("biosemi64")
    data.set_montage(montage)

    data, events = data.resample(preprocessing_kwargs['sfreq'], events=events)

    # Applying GEDAI artifact cleaning
    from gedai import Gedai
    gedai = Gedai()
    gedai.fit_raw(
        data,
        verbose=True,
    )
    data = gedai.transform_raw(
        data, verbose=False
    )
    return data, events

preprocessing = dict(sfreq=256, #Resampling the data at this sampling frequency
                     lowpass=40, # at which frequency we want to apply the lowpass filter
                     highpass=.5, #Same for highpass, see FAQ
                     reference='average', #What reference shold the data be re-referenced to
                     pick_channels=['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5',
                        'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3',
                        'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz',
                        'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz',
                        'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz',
                        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2',
                        'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
                    )

# Reading the data
epoch_data, info = hmp.io.read_bids_raw(
    bids_kwargs=bids_kwargs,
    epoching_kwargs=epoching,
    preprocessing_kwargs=preprocessing,
    centering_id = centering_id,
    event_id = event_id,
    montage='biosemi64',#What MNE built-in montage should be used for that data
    # Note that this can also be an mne.DigMontage in case there are no standard montage for your data
    preprocessing_fn=mypreprocessing,
    cpus=1, #Can also read data in parallel
    verbose = True, # For the documentation only we turn verbose off to avoid cluttered outputs, remove for other purposes
)

# Saving
epoch_data.to_netcdf('../estimation_files/epoch_data.nc')
info.save('../estimation_files/info.fif', overwrite=True)
print(epoch_data)

