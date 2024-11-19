import os

DATA_DIR = os.path.realpath(__file__).replace(os.path.join('utils', 'constants.py'), 'database')
TREATMENT_DATA = os.path.join(DATA_DIR, 'RTdata_BasedOnMRNlist_DeId.xlsx')
APPOINTMENT_DATA = os.path.join(DATA_DIR, 'first-RT-appointment-jan2021-apr2022_DeId.xlsx')