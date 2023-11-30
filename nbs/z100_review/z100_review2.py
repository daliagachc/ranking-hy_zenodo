# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# region imports
from IPython import get_ipython

# noinspection PyBroadException
try:
    _magic = get_ipython().run_line_magic
    _magic("load_ext", "autoreload")
    _magic("autoreload", "2")
except:
    pass

# noinspection PyUnresolvedReferences
import datetime as dt

# noinspection PyUnresolvedReferences
import glob

# noinspection PyUnresolvedReferences
import os

# noinspection PyUnresolvedReferences
import pprint

# noinspection PyUnresolvedReferences
import sys

# noinspection PyUnresolvedReferences
import cartopy as crt

# noinspection PyUnresolvedReferences
import matplotlib as mpl

# noinspection PyUnresolvedReferences
import matplotlib.colors

# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt

# noinspection PyUnresolvedReferences
import numpy as np

# noinspection PyUnresolvedReferences
import pandas as pd

# noinspection PyUnresolvedReferences
import seaborn as sns

# noinspection PyUnresolvedReferences
import xarray as xr

# noinspection PyUnresolvedReferences
# import bnn_tools.bnn_array

plt.style.use("default")
xr.set_options(
    display_expand_data=False,
    display_expand_data_vars=True,
    display_max_rows=10,
    display_style="html",
    display_width=80,
    display_expand_attrs=False,
)

import funs_ as fu
# endregion

# %%
P = "/Users/aliaga/Documents/Work_DA/Py-packs/ranking-hy/data/d1/ds_5.nc"

# %%
DS = xr.open_dataset(P)

# %%
fu.plot_a(DS)

# %%
fu.plot_b(DS)

# %%
fu.plot_d(DS)

# %%
fu.plot_e(DS)

# %%
fu.plot_f(DS)

# %%
fu.plot_g(DS)

# %%
fu.plot_h(DS)

# %%
fu.plot_i(DS)

# %%
for m in ["mean", "median", "max"]:
    SE = fu.get_se(DS, m)
    fu.plot_below(m, SE)
    plt.show()

# %%
(SE["mTR"].to_dataframe()[["mTR", "q"]].corr(method="spearman"))


# %%
for mm in ["max", "median", "mean"]:
    fu.plot_j(DS, mm)

# %%
(SE["n_max_r"].to_dataframe()[["n_max_r", "q"]].corr(method="spearman"))

# %%
fu.plot_k(SE)

# %%
fu.plot_l(DS)

# %%
fu.plot_m(DS)

# %%
fu.plot_nn(DS)

# %%
