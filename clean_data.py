import pandas as pd
import numpy as np

def add_language_level(df):
    df.loc[(df['WPOTAALTV'] == 1), 'WPOTAALTV'] = 0
    df.loc[(df['WPOTAALTV'] == 2), 'WPOTAALTV'] = 1
    df.loc[(df['WPOTAALTV'] == 4), 'WPOTAALTV'] = 2
    df.loc[(df['WPOTAALLV'] == 1), 'WPOTAALLV'] = 0
    df.loc[(df['WPOTAALLV'] == 2), 'WPOTAALLV'] = 1
    df.loc[(df['WPOTAALLV'] == 4), 'WPOTAALLV'] = 2
    return df

def add_level_3yrs(df):
    df['LEVEL3'] = np.nan
    df.loc[((df['ONDERW_DETAILVO'] == 1) | (df['ONDERW_DETAILVO'] == 2) | (df['ONDERW_DETAILVO'] == 20)), 'LEVEL3'] = 0
    df.loc[((df['ONDERW_DETAILVO'] == 5) | (df['ONDERW_DETAILVO'] == 6) | (df['ONDERW_DETAILVO'] == 7)), 'LEVEL3'] = 1
    df.loc[(df['ONDERW_DETAILVO'] == 3), 'LEVEL3'] = 2
    df.loc[(df['ONDERW_DETAILVO'] == 8), 'LEVEL3'] = 3
    df.loc[(df['ONDERW_DETAILVO'] == 4), 'LEVEL3'] = 4
    df.loc[(df['ONDERW_DETAILVO'] == 9), 'LEVEL3'] = 5
    df.loc[(df['ONDERW_DETAILVO'] == 10), 'LEVEL3'] = 7
    df.loc[(df['ONDERW_DETAILVO'] == 11), 'LEVEL3'] = 9
    df.loc[((df['ONDERW_DETAILVO'] == 12) | (df['ONDERW_DETAILVO'] == 13) | (df['ONDERW_DETAILVO'] == 14)), 'LEVEL3'] = 11
    return df

def add_advice_final(df):
    df['ADV_FINAL'] = df['ADV_TEACH'].copy()
    df.loc[((df['WPOADVIESHERZ'] == 0) | (df['WPOADVIESHERZ'] == 1) | (df['WPOADVIESHERZ'] == 10) | (df['WPOADVIESHERZ'] == 80)), 'ADV_FINAL'] = 0
    df.loc[((df['WPOADVIESHERZ'] == 20) | (df['WPOADVIESHERZ'] == 21)), 'ADV_FINAL'] = 1
    df.loc[((df['WPOADVIESHERZ'] == 22) | (df['WPOADVIESHERZ'] == 23)), 'ADV_FINAL'] = 2
    df.loc[((df['WPOADVIESHERZ'] == 30) | (df['WPOADVIESHERZ'] == 31)), 'ADV_FINAL'] = 3
    df.loc[((df['WPOADVIESHERZ'] == 32) | (df['WPOADVIESHERZ'] == 33)), 'ADV_FINAL'] = 4
    df.loc[((df['WPOADVIESHERZ'] == 40) | (df['WPOADVIESHERZ'] == 41) | (df['WPOADVIESHERZ'] == 34) | (df['WPOADVIESHERZ'] == 35)), 'ADV_FINAL'] = 5
    df.loc[((df['WPOADVIESHERZ'] == 42) | (df['WPOADVIESHERZ'] == 43)), 'ADV_FINAL'] = 6
    df.loc[((df['WPOADVIESHERZ'] == 50) | (df['WPOADVIESHERZ'] == 51) | (df['WPOADVIESHERZ'] == 44) | (df['WPOADVIESHERZ'] == 45)), 'ADV_FINAL'] = 7
    df.loc[(df['WPOADVIESHERZ'] == 52), 'ADV_FINAL'] = 8
    df.loc[(df['WPOADVIESHERZ'] == 60), 'ADV_FINAL'] = 9
    df.loc[(df['WPOADVIESHERZ'] == 61), 'ADV_FINAL'] = 10
    df.loc[(df['WPOADVIESHERZ'] == 70), 'ADV_FINAL'] = 11
    return df

def add_advice_test(df):
    df['ADV_TEST'] = np.nan
    df.loc[(df['WPOTOETSADV'] == 10), 'ADV_TEST'] = 0
    df.loc[((df['WPOTOETSADV'] == 11) | (df['WPOTOETSADV'] == 20)), 'ADV_TEST'] = 1
    df.loc[(df['WPOTOETSADV'] == 22), 'ADV_TEST'] = 2
    df.loc[(df['WPOTOETSADV'] == 30), 'ADV_TEST'] = 3
    df.loc[(df['WPOTOETSADV'] == 34), 'ADV_TEST'] = 5
    df.loc[(df['WPOTOETSADV'] == 42), 'ADV_TEST'] = 6
    df.loc[(df['WPOTOETSADV'] == 44), 'ADV_TEST'] = 7
    df.loc[(df['WPOTOETSADV'] == 60), 'ADV_TEST'] = 9
    df.loc[(df['WPOTOETSADV'] == 61), 'ADV_TEST'] = 10
    df.loc[(df['WPOTOETSADV'] == 70), 'ADV_TEST'] = 11
    return df

def add_advice_teacher(df):
    df['ADV_TEACH'] = np.nan
    df.loc[((df['WPOADVIESVO'] == 0) | (df['WPOADVIESVO'] == 1) | (df['WPOADVIESVO'] == 10) | (df['WPOADVIESVO'] == 80)), 'ADV_TEACH'] = 0
    df.loc[((df['WPOADVIESVO'] == 20) | (df['WPOADVIESVO'] == 21)), 'ADV_TEACH'] = 1
    df.loc[((df['WPOADVIESVO'] == 22) | (df['WPOADVIESVO'] == 23)), 'ADV_TEACH'] = 2
    df.loc[((df['WPOADVIESVO'] == 30) | (df['WPOADVIESVO'] == 31)), 'ADV_TEACH'] = 3
    df.loc[((df['WPOADVIESVO'] == 32) | (df['WPOADVIESVO'] == 33)), 'ADV_TEACH'] = 4
    df.loc[((df['WPOADVIESVO'] == 40) | (df['WPOADVIESVO'] == 41) | (df['WPOADVIESVO'] == 34) | (df['WPOADVIESVO'] == 35)), 'ADV_TEACH'] = 5
    df.loc[((df['WPOADVIESVO'] == 42) | (df['WPOADVIESVO'] == 43)), 'ADV_TEACH'] = 6
    df.loc[((df['WPOADVIESVO'] == 50) | (df['WPOADVIESVO'] == 51) | (df['WPOADVIESVO'] == 44) | (df['WPOADVIESVO'] == 45)), 'ADV_TEACH'] = 7
    df.loc[(df['WPOADVIESVO'] == 52), 'ADV_TEACH'] = 8
    df.loc[(df['WPOADVIESVO'] == 60), 'ADV_TEACH'] = 9
    df.loc[(df['WPOADVIESVO'] == 61), 'ADV_TEACH'] = 10
    df.loc[(df['WPOADVIESVO'] == 70), 'ADV_TEACH'] = 11
    return df

def add_ethnicity(df):
    df['ETHN'] = np.nan
    df.loc[(df['LANDAKTBO'] == 2), 'ETHN'] = 4
    df.loc[(df['LANDAKTBO'] == 3), 'ETHN'] = 5
    df.loc[((df['LANDAKTBO'] == 1) | ((df['LANDAKTBO'] >= 4) & (df['LANDAKTBO'] <= 9))), 'ETHN'] = 6
    df.loc[df['ETHGRP'] == 0, 'ETHN'] = 0
    df.loc[df['ETHGRP'] == 1, 'ETHN'] = 1
    df.loc[df['ETHGRP'] == 2, 'ETHN'] = 3
    df.loc[((df['ETHGRP'] == 3) | (df['ETHGRP'] == 4)), 'ETHN'] = 2
    return df

def add_WOZ(df):
    # control for increase WOZ houses each year
    # schoolyear 2015 YYYY = 2016
    # schoolyear 2016 YYYY = 2017
    # schoolyear 2017 YYYY = 2018
    mean_16=208000
    mean_17=217000
    mean_18=230000
    inc_17 = (mean_17 - mean_16) / mean_16
    inc_18 = (mean_18 - mean_16) / mean_16
    df['WOZ'] = df.apply(lambda x: (x['WOZ'] / (inc_17 + 1)) if x['SCHOOLYEAR'] == 2016 else x['WOZ'], axis=1)
    df['WOZ'] = df.apply(lambda x: (x['WOZ'] / (inc_18 + 1)) if x['SCHOOLYEAR'] == 2017 else x['WOZ'], axis=1)
#     print(df.groupby('SCHOOLYEAR')['WOZ'].mean())
#     SCHOOLYEAR
#     2015    259980.831516
#     2016    259876.033432
#     2017    255281.347083
#     winsorize WOZ (extreme values)
#     print(df[df['WOZ'] >= 2000000].shape[0]) #466
#     print(df[df['WOZ'] >= 2000000].shape[0] / len(df)) #0.000876795219396366
#     print("")
#     print("Amount of zeros in WOZ")
#     print(df[df['WOZ'] == 0].shape[0]) #3253
#     print("Amount above 5 million WOZ")
#     print(df[df['WOZ'] >= 5000000].shape[0]) # 57
    df.loc[(df['WOZ'] >= 5000000), 'WOZ'] = 5000000
#     print("")
#     print("Winsorized mean:")
#     print(df.groupby('SCHOOLYEAR')['WOZ'].mean())
#     Winsorized mean:
#     SCHOOLYEAR
#     2015    259640.133939
#     2016    259392.611274
#     2017    254999.683216
    return df

def handle_NaNs(df_clean):
    with open('NaNs.txt', 'a') as f:
        print("", file=f)
        print("All NaNs after SO selection:", file=f)
        print(df_clean.isna().sum(), file=f)
        print(df_clean.shape, file=f)
    df_clean.dropna(subset = ['GENDER', 'ETHN', 'WPOTAALLV', 'WPOTAALTV', 'ADV_TEACH', 'ADV_TEST', 'ADV_FINAL', 'LEVEL3', 'VOLEERJAAR'], inplace=True)
    df_clean = df_clean.drop(columns='VOLEERJAAR')
    with open('NaNs.txt', 'a') as f:
        print("", file=f)
        print("NaNs in SES:", file=f)
        print(df_clean.loc[:, ['PERCBESTINKO', 'OPLNIVMA', 'OPLNIVPA', 'SECMMA', 'SECMPA', 'WOZ']].isna().sum(axis=1).value_counts(), file=f)
        print(df_clean.shape, file=f)

def clean_data(df):
    df = add_ethnicity(df)
    df = add_advice_teacher(df)
    df = add_advice_test(df)
    df = add_advice_final(df)
    df = add_level_3yrs(df)
    df = add_language_level(df)
    df = add_WOZ(df)
    df = df.rename(columns={"GESLACHT": "GENDER"})
    df['GENDER'] -= 1
    df['DD'] = 1
    with open('NaNs.txt', 'w+') as f:
        print("All NaNs before SO selection:", file=f)
        print(df[['GENDER', 'ETHN', 'WPOTAALLV', 'WPOTAALTV', 'ADV_TEACH', 'ADV_TEST', 'ADV_FINAL', 'LEVEL3', 'WOZ', 'PERCBESTINKO', 'OPLNIVMA', 'OPLNIVPA', 'SECMMA', 'SECMPA', 'VOLEERJAAR']].isna().sum(), file=f)
        print(df.shape, file=f)
    df_clean = df[['GENDER', 'ETHN', 'WPOTAALLV', 'WPOTAALTV', 'ADV_TEACH', 'ADV_TEST', 'ADV_FINAL', 'LEVEL3', 'WOZ', 'PERCBESTINKO', 'OPLNIVMA', 'OPLNIVPA', 'SECMMA', 'SECMPA', 'VOLEERJAAR', 'YYYY', 'MM', 'DD']].copy()
    df_clean = df_clean.drop(df_clean[(df_clean.VOLEERJAAR == 1) | (df_clean.VOLEERJAAR == 2) | (df_clean.VOLEERJAAR == 4) | (df_clean.VOLEERJAAR == 5) | (df_clean.VOLEERJAAR == 6) | (df_clean.VOLEERJAAR == 9)].index)
    df_clean = df_clean.drop(df_clean[df_clean.ADV_TEACH == 0].index)
    df_clean = df_clean.drop(df_clean[df_clean.ADV_TEST == 0].index)
    df_clean = df_clean.drop(df_clean[df_clean.ADV_FINAL == 0].index)
    df_clean = df_clean.drop(df_clean[df_clean.LEVEL3 == 0].index)
    handle_NaNs(df_clean)
    return df_clean
