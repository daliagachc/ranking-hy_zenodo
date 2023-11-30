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

import mod.funs as fu

plt.style.use("default")
xr.set_options(
    display_expand_data=False,
    display_expand_data_vars=True,
    display_max_rows=10,
    display_style="html",
    display_width=80,
    display_expand_attrs=False,
)
# endregion


def main():

    # %%
    d1 = import_class_ds()

    # %%
    N = "$N_{2.5-5\mathrm{nm}}$"
    ds = xr.open_dataset("../../data/d1/ds_5.nc").rename({"N": N})

    res = merge_ds_class(ds, d1)

    # %%

    plot_hist(res)

    # %%


def merge_ds_class(
    ds, d1, new_name_class="Classification", new_name_perc_ranking="Percentile ranking"
):
    d5 = ds.pipe(lambda d: xr.merge([d, d1])).squeeze()
    npf_class = "npf_class"
    q_col_name = "q"
    res = (
        d5[[q_col_name, npf_class]]
        .to_dataframe()
        .dropna(how="all")
        .rename({npf_class: new_name_class}, axis=1)
        .rename({q_col_name: new_name_perc_ranking}, axis=1)
    )
    return res


def plot_hist(
    res,
    name_fig_out="f040.pdf",
    name_perc_ranking="Percentile ranking",
    name_class="Classification",
    x_label="Percentile ranking ($\Delta N_{2.5-5\mathrm{nm}}$)",
):
    # region sciplot
    # noinspection PyUnresolvedReferences
    import scienceplots

    plt.style.use(
        [
            "default",
            "acp",
            # 'notebook',
            # 'sp-grid',
            "no-black",
            "illustrator-safe",
        ]
    )
    # endregion sciplot
    sns.histplot(
        data=res,
        x=name_perc_ranking,
        hue=name_class,
        multiple="stack",
        bins=np.arange(0, 101, 5),
        # palette='tab10',
        # palette='Set2',
        palette="RdYlBu_r",
        hue_order=pd.Series(
            ["event Ia", "event Ib", "event II", "undefined", "non event"]
        )[::-1],
        # order = ord_
        alpha=1,
    )
    ax: plt.Axes = plt.gca()
    ax.set_xlim(0, 100)
    ax.set_box_aspect(1 / 1.6)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.1, 1.3))
    ax.set_xlabel(x_label)
    f: plt.Figure = plt.gcf()
    a_ = 2.5
    f.set_size_inches(a_ + 1, a_ / 1.6)
    import uscit

    uscit.set_margin(f, x1=0.5, x2=1.4, y1=0.5)
    # uscit.set_fig_rule(f)
    # region temp open fig
    _p_ = name_fig_out
    plt.gcf().savefig(_p_, transparent=True)
    os.system(f"open {_p_}")
    # endregion temp open fig


def import_class_ds(name_class_col="npf_class"):
    cols = [
        "Matlab datenum",
        "event Ia",
        "event Ib",
        "event II",
        "event Apple",
        "event Bump",
        "event Rain",
        "event Featureless",
        "non event",
        "undefined",
        "bad data",
        "partly bad data",
        "checksum",
    ]
    p = "../../data/data_orig/DMPS_Event_Classification_1996-2022.txt"
    d = (
        pd.read_csv(p, names=cols, comment="%", sep="\s+")
        .assign(
            day=lambda d: pd.to_datetime(d["Matlab datenum"] - 719529, unit="D").round(
                "10T"
            )
        )
        .set_index("day")["2018-01-01":]
        .drop("Matlab datenum", axis=1)
        .drop("checksum", axis=1)
        .drop("bad data", axis=1)
        # [['event Ib','event II','bad data','non event', 'undefined']]
    )
    d1 = (
        d.stack()
        .where(lambda d: d > 0)
        .dropna()
        .reset_index()
        .drop_duplicates("day")
        .set_index("day")
        .rename({"level_1": name_class_col}, axis=1)[name_class_col]
        .to_xarray()
    )
    return d1
