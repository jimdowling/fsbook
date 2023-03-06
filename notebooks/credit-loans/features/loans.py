import pandas as pd
import datetime
import numpy as np

def home_ownership_udf(home_ownership: str)-> str:
    if (home_ownership == 'ANY' or home_ownership == 'NONE') :
        return 'OTHER'
    return home_ownership


def pub_rec(number):
    if number == 0.0:
        return 0
    else:
        return 1


def mort_acc(number):
    if number == 0.0:
        return 0
    elif number >= 1.0:
        return 1
    else:
        return number


def pub_rec_bankruptcies(number):
    if number == 0.0:
        return 0
    elif number >= 1.0:
        return 1
    else:
        return number


def fill_mort_acc(total_acc, mort_acc, total_acc_avg):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc].round()
    else:
        return mort_acc


def extract_zipcode(address):
    return address[-5:]


def extract_year(earliest_cr_line):
    return earliest_cr_line.year

