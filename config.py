import os.path

PROJECT_ROOT = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
INITIAL_DATA_PATH = os.path.join(DATA_DIR, 'HH_Provider_Oct2023.csv')
PCA_PATH = os.path.join(DATA_DIR, 'pca.csv')
CLEAN_DATA_PATH = os.path.join(DATA_DIR, 'clean_data.csv')
TRANSFORMED_DATA_PATH = os.path.join(DATA_DIR, 'transformed.csv')
TARGET_COLUMN_NAME = 'Quality of patient care star rating'
NN_SAVE_PATH = os.path.join(PROJECT_ROOT, 'NN', 'net.pkl')

QUANTIFIABLE_COLUMNS = {
    "Offers Nursing Care Services",
    "Offers Physical Therapy Services",
    "Offers Occupational Therapy Services",
    "Offers Speech Pathology Services",
    "Offers Medical Social Services",
    "Offers Home Health Aide Services",
    "Certification Date",
    "Quality of patient care star rating",
    "How often the home health team began their patients' care in a timely manner",
    "How often the home health team determined whether patients received a flu shot for the current flu season",
    "How often patients got better at walking or moving around",
    "How often patients got better at getting in and out of bed",
    "How often patients got better at bathing",
    "How often patients' breathing improved",
    "How often patients got better at taking their drugs correctly by mouth",
    "How often home health patients had to be admitted to the hospital",
    "How often patients receiving home health care needed urgent, unplanned care in the ER without being admitted",
    "Changes in skin integrity post-acute care: pressure ulcer/injury",
    "How often physician-recommended actions to address medication issues were completely timely",
    "Percent of Residents Experiencing One or More Falls with Major Injury",
    "Application of Percent of Long Term Care Hospital Patients with an Admission and Discharge Functional Assessment and a Care Plan that Addresses Function",
    "DTC Numerator",
    "DTC Denominator",
    "DTC Observed Rate",
    "DTC Risk-Standardized Rate",
    "DTC Risk-Standardized Rate (Lower Limit)",
    "DTC Risk-Standardized Rate (Upper Limit)",
    "PPR Numerator",
    "PPR Denominator",
    "PPR Observed Rate",
    "PPR Risk-Standardized Rate",
    "PPR Risk-Standardized Rate (Lower Limit)",
    "PPR Risk-Standardized Rate (Upper Limit)",
    "PPH Numerator",
    "PPH Denominator",
    "PPH Observed Rate",
    "PPH Risk-Standardized Rate",
    "PPH Risk-Standardized Rate (Lower Limit)",
    "PPH Risk-Standardized Rate (Upper Limit)",
    "How much Medicare spends on an episode of care at this agency, compared to Medicare spending across all agencies nationally",
    "No. of episodes to calc how much Medicare spends per episode of care at agency, compared to spending at all agencies (national)"
}

RANDOM_STATE = 42
EPOCHS = 15000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001
