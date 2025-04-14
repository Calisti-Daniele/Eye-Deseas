from scipy.fftpack import fft
from scipy.stats import linregress

from functions.utility_functions import *
import numpy as np


def fourier_transform(row):
    return np.abs(fft(row.values))[1]  # Prendiamo il primo coefficiente utile

def compute_trend(row):
    x = np.arange(len(row))
    y = row.values
    slope, _, _, _, _ = linregress(x, y)
    return slope


df = read_csv('../../training/xgboost/feature_selection/train_dataset_trex_not_augmented_normalized.csv')

df["fourier_coeff"] = df.apply(fourier_transform, axis=1)
df["trend"] = df.apply(compute_trend, axis=1)

save_csv(df, "../../training/xgboost/feature_selection/train_dataset_trex_not_augmented_normalized.csv")
