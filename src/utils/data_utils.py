import enum


class EncounterSet(enum.Enum):
    PROCESSED = "processed"
    SUFFICIENT = "sufficient"
    CENSORED = "censored"
    LOS = "los"
    MORTALITY = "mortality"
    READMISSION = "readmission"


class Split(enum.Enum):
    TRAIN = "train"
    TEST = "test"
    VAL = "val"


OUTCOME_ENCOUNTERS = [
    EncounterSet.LOS,
    EncounterSet.MORTALITY,
    EncounterSet.READMISSION,
]


class DatasetGenerationJob(enum.Enum):
    ALL_PROCESSED = (
        "all_processed"  # generate data from processed encounters, all x-x-xs
    )
    ALL_SUFFICIENT = (
        "all_sufficient"  # generate data from sufficient encounters, all x-x-xs
    )
    LOS = "los"  # generate data from LOS outcome encounters, all x-x-xs
    MORTALITY = (
        "mortality"  # generate data from MORTALITY outcome encounters, all x-x-xs
    )
    READMISSION = (
        "readmission"  # generate data from READMISSION outcome encounters, all x-x-xs
    )


OUTCOME_JOB_TO_ENCOUNTER_SET = {
    DatasetGenerationJob.LOS: EncounterSet.LOS,
    DatasetGenerationJob.MORTALITY: EncounterSet.MORTALITY,
    DatasetGenerationJob.READMISSION: EncounterSet.READMISSION,
}
