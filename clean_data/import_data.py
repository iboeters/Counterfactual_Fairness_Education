import pandas as pd
import numpy as np
import os
import sys

def read_data(file_name, dir_datasets):
    cwd = os.getcwd()
    colspecs = [(0, 1), (1, 10), (10, 11), (11, 15), (15, 17), (17, 21),  (21, 22), (22, 23), (23, 27), (27, 28), (28, 29), (29, 33), (33, 34), (34, 45), (45, 48), (48, 60), (60, 72), (72, 75), (75, 78), (78, 80), (80 ,82), (82 ,86), (86, 88), (88, 90), (90, 92), (92, 95), (95, 98), (98, 105), (105, 107), (107, 109), (109, 112), (112, 114), (114, 116), (116, 117), (117, 118), (118, 121), (121, 125), (125, 127), (127, 130), (130, 131), (131, 133), (133, 137), (137, 141)]
    var_names = ["RINPERSOONS", "RINPERSOON", "GESLACHT", "YYYY", "MM", "HERGR", "GEN", "NOUDER", "LANDAKT", "ETHGRP", "LANDAKTBO", "POSTCO", "STEDGEM", "WOZ", "PERCBESTINKO", "GEWOPLPA", "GEWOPLMA", "OPLNIVPA", "OPLNIVMA", "SECMPA", "SECMMA", "WPOBRIN", "WPOBRINVEST", "WPOADVIESVO", "WPOCODEEINDTOETS", "CITOGOEDTAAL", "CITOPERCTAAL", "CITOZSCORETAAL", "WPOTAALLV", "WPOTAALTV", "WPOUITSLEINDT", "WPOTOETSADV", "WPOADVIESHERZ", "INSCHRWPO", "INSCHRWEC", "ONDERW_1", "BRINHOOFD", "VOBRINVEST", "ONDERW_3", "VOLEERJAAR", "ONDERW_DETAILVO", "POSTCOBRINVO", "POSTCODEBRINWPOWEC"]
    df = pd.read_fwf(cwd + "/" + dir_datasets + file_name, colspecs=colspecs, names=var_names)
    return df

def import_data():
    dir_datasets =  "/datasets/"
    file_name_15 = "SCHOOLADVIES2015ANAV1.ASC"
    file_name_16 = "SCHOOLADVIES2016ANAV1.ASC"
    file_name_17 = "SCHOOLADVIES2017ANAV1.ASC"

    df_15 = read_data(file_name_15, dir_datasets)
    df_15['SCHOOLYEAR'] = 2015
    df_16 = read_data(file_name_16, dir_datasets)
    df_16['SCHOOLYEAR'] = 2016
    df_17 = read_data(file_name_17, dir_datasets)
    df_17['SCHOOLYEAR'] = 2017
    df = pd.concat([df_15, df_16, df_17])
    return df
