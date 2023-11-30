# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---


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
# endregion


def set_margin(f=None, x1=None, x2=None, y1=None, y2=None):
    if f is None:
        f: plt.Figure = plt.gcf()
    bbox = f.get_window_extent().transformed(f.dpi_scale_trans.inverted())
    x, y = bbox.width, bbox.height
    # x,y = f.get_size_inches()
    # print(x, y)
    if x1 is not None:
        f.subplots_adjust(left=x1 / x)
    if x2 is not None:
        f.subplots_adjust(right=1 - x2 / x)
    if y1 is not None:
        f.subplots_adjust(bottom=y1 / y)
    if y2 is not None:
        # print(y2)
        f.subplots_adjust(top=1 - y2 / y)


def plot_a(ds):
    (
        ds.assign_coords({"hr": lambda d: d["hour"].round()})[{"Dp": slice(5, 20, 2)}][
            "dndlDp"
        ]
        .groupby("hr")
        .median()
        .loc[{"hr": slice(0, 30)}]
        .where(lambda d: d["g"] != "", drop=True)
        .groupby("g")
        .median()
        .loc[{"id": "nd"}]
        .plot(
            col="g",
            row="Dp",
            hue="Dp",
            yscale="log",
            aspect=2,
            size=2,
        )
        # ['g'].to_series()
    )


def plot_b(ds):
    # global df, df1, ax
    df = (
        get_ds_nais_roll_loc_dp_med(ds)
        # .assign_coords({'hm':lambda d:d.idxmax('hour')})
        # .assign_coords({'max':lambda d:d.max('hour')})
        .idxmax("hour").to_series()
        # .reset_index()
        # ['g'].to_series()
    )

    df1 = df.unstack("g")
    # df1

    sciplot_def()
    df = (
        get_ds_nais_roll_loc_dp_med(ds).plot(
            col="g",
            yscale="log",
            norm=mpl.colors.LogNorm(
                # vmin=1000,vmax=10000
            ),
            # cmap='plasma'
            size=1.5,
            aspect=1,
        )
        # ['g'].to_series()
    )
    for ax in df.axs.flat:
        ax.set_box_aspect(1 / 1.6)
        ax.set_xticks(np.arange(12, 30, 6))
    for ax, g in zip(df.axs.flat, ["g1", "g2", "g3"]):
        ax.scatter(x=df1[g], y=df1.index, s=1, c=".2")


def get_ds_nais_roll_loc_dp_med(ds):
    return (
        ds.loc[{"id": "nais"}]["dndlDp"]
        .rolling({"hour": 3}, center=True)
        .median()
        .loc[{"hour": slice(8, 30)}]
        .loc[{"Dp": slice(1e-9, 40e-9)}]
        .where(lambda d: d["g"] != "", drop=True)
        .groupby("g")
        .median()
    )


def sciplot_def(grid=True):
    # region sciplot
    # noinspection PyUnresolvedReferences
    import scienceplots

    if grid:
        sp_grid = ["sp-grid"]
    else:
        sp_grid = []
    plt.style.use(
        [
            "default",
            "acp",
            # 'notebook',
            *sp_grid,
            "latex-sans",
            "no-black",
            "illustrator-safe",
        ]
    )
    # endregion sciplot


def plot_d(ds):
    # global dp, n, q, df, res, ax, d_
    sciplot_def()
    dp = "$D_p$ [nm]"
    n = "N [cm$^{-3}$]"
    q = "P [\%]"
    df = (
        ds.loc[{"id": "nais"}]["dndlDp"]
        .rolling({"hour": 5}, center=True)
        .median()
        .loc[{"hour": slice(8, 30)}]
        .loc[{"Dp": slice(1e-9, 27e-9)}]
        .where(lambda d: d["qCut"] != "nan", drop=True)
        # .assign_coords({'hm':lambda d:d.idxmax('hour')})
        # .assign_coords({'max':lambda d:d.max('hour')})
        .max("hour")
        .pipe(lambda d: d * 0.05)
        .rename(n)
        .reset_coords()[["q", n]]
        .assign({dp: lambda d: (d["Dp"] * 1e9).round(1)})
        .assign({q: lambda d: (d["q"] / 5).round() * 5})
        .to_dataframe()
        # .to_series()
        # .reset_index()
        # ['g'].to_series()
    )
    # df

    dps = df[dp].unique()
    dps.sort()

    sciplot_def()
    res = sns.relplot(
        data=df,
        x=q,
        y=n,
        col=dp,
        col_wrap=5,
        height=1.4,
        # marker='.',
        # alpha=.2,
        # s=10,
        kind="line",
        estimator="median",
        errorbar=("pi", 50)
        # sharey=True
    )
    for ax in res.axes.flat:
        ax.set_yscale("log")
        # ax.set_ylim(.3e2,.2e4)
        ax.set_box_aspect(1 / 1.6)
    for ax, d_ in zip(res.axes.flat, dps):
        dd = df.where(lambda d: d[dp] == d_).dropna()
        ax.scatter(x=dd["q"], y=dd[n], marker=",", lw=0.5, s=1, alpha=0.2, c="C1")

    # region temp open fig
    _p_ = "fig_d.pdf"
    plt.gcf().savefig(_p_, transparent=True)
    os.system(f"open {_p_}")
    # endregion temp open fig


def plot_e(ds):
    sciplot_def()
    # global res, ax
    dp = "$D_p$ [nm]"
    # n = 'N [cm$^{-3}$]'
    q = "P [\%]"
    df = (
        ds.loc[{"id": "nd"}]["dndlDp"]
        .rolling({"hour": 5}, center=True)
        .median()
        .loc[{"Dp": slice(1e-9, 27e-9)}]
        .where(lambda d: d["qCut"] != "nan", drop=True)
        # .assign_coords({'hm':lambda d:d.idxmax('hour')})
        # .assign_coords({'max':lambda d:d.max('hour')})
        .loc[{"hour": slice(10, 25)}]
        .idxmax("hour")
        # .loc[{'hour':slice(9,25)}]
        .assign_coords({dp: lambda d: (d["Dp"] * 1e9)})
        .assign_coords({q: lambda d: (d["q"] / 5).round() * 5})
        .to_dataframe()
        .reset_index()
        .sort_values("q")
    )

    # qu = df.sort_values('q')['qCut'].drop_duplicates()

    res = sns.relplot(
        data=df,
        x="hour",
        y=dp,
        orient="y",
        kind="line",
        estimator="median",
        errorbar=("pi", 50),
        col="qCut",
        col_wrap=4,
        height=1.2,
        facet_kws=dict(despine=False),
        aspect=1.2,
    )
    for ax in res.axes.flat:
        ax.set_yscale("log")
        t = ax.get_title()
        ax.set_title(t.replace(".0", "").replace("qCut = ", ""))
        ax.set_box_aspect(1 / 1.6)
        ax.set_xlim(9, 24)
        ax.set_xticks(np.arange(12, 25, 6))
        ax.set_xticks(np.arange(10, 25, 2), minor=True)
    # for ax,d_ in zip(res.axes.flat,qu):
    #     dd=df.where(lambda d:d['qCut']==d_).dropna()
    # ax.scatter(x=dd['hour']+np.random.randn(len(dd))*.2,y=dd[dp],marker='.',s=10,alpha=.05,c='C1')
    # region temp open fig
    _p_ = "fig_e.pdf"
    plt.gcf().savefig(_p_, transparent=True)
    os.system(f"open {_p_}")
    # endregion temp open fig


def plot_f(ds):
    (ds.loc[{"id": "nd"}].sum("Dp").groupby("g").median()["dndlDp"].plot(col="g"))


def plot_g(ds):
    (
        ds
        # .loc[{'id':'dmps'}]
        .median("hour")
        .median("day")["dndlDp"]
        .plot(xscale="log", yscale="log", hue="id")
    )


def plot_h(ds):
    (
        ds.assign_coords({"s": lambda d: d["day"].dt.season})
        .assign_coords({"hr": lambda d: d["hour"].round()})
        .groupby("hr")
        .median()
        .loc[{"hr": [0, 4, 8, 12, 16, 20]}]
        .groupby("s")
        .median()
        # .groupby('s').median()
        # .loc[{'id':'dmps'}]
        # .median('hour')
        # .median('day')
        ["dndlDp"]
        .plot(
            xscale="log",
            yscale="log",
            hue="id",
            col="s",
            row="hr",
            size=1.5,
            aspect=1.6,
        )
    )


def plot_i(ds):
    # global se, ax
    se = (
        ds.loc[{"id": "nd"}]["dndlDp"]
        .sum("Dp")
        .rolling({"hour": 6}, center=True)
        .median()
        .where(lambda d: d["g"] != "", drop=True)
        .to_dataframe()
        .reset_index()
    )

    res = sns.relplot(
        data=se,
        x="hour",
        y="dndlDp",
        col="g",
        kind="line",
        units="day",
        estimator=None,
        alpha=0.1,
        height=2,
    )
    for ax in res.axes.flat:
        ax.set_yscale("log")
        ax.set_xlim(-6, 30)
        ax.set_xticks([0, 12, 24])


def plot_j(DS, mm):
    # global SE, ax, _p_
    sciplot_def(grid=False)
    nn = f"n_{mm}_r"
    max_n____ = f"Ranking\n{mm} $ N_{{2.5-5}}$"
    f__pdf = f"f02_{mm}.pdf"
    SE = get_se(DS, mm)
    sns.scatterplot(
        data=SE[nn].to_dataframe(),
        y=nn,
        x="q",
        s=5,
        alpha=0.3,
        hue="g",
        palette=["C1", "C2", "C4"],
        # scatter = False
    )
    # noinspection DuplicatedCode
    ax: plt.Axes = plt.gca()
    f: plt.Figure = plt.gcf()
    f.set_size_inches(3, 2)
    ax.set_box_aspect(1)
    set_margin(f, x1=0.7, y1=0.5, x2=1)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax = plt.gca()
    ax.get_legend().set_bbox_to_anchor([1, 1])
    ax.set_xlabel("Ranking $\Delta N_{2.5-5}$")
    ax.set_ylabel(max_n____)
    _p_ = f__pdf
    plt.gcf().savefig(_p_, transparent=True)
    plt.show()


def plot_k(SE):

    sciplot_def()
    sns.scatterplot(
        data=SE[["n_max", "d_lN"]].to_dataframe(),
        y="n_max",
        x="d_lN",
        s=5,
        alpha=0.3,
        hue="g",
        palette=["C1", "C2", "C4"],
        # scatter = False
    )
    ax: plt.Axes = plt.gca()
    ax.set_yscale("log")
    _p_ = "f03.pdf"
    plt.gcf().savefig(_p_, transparent=True)


def plot_l(DS):
    (
        DS.loc[{"id": "nd"}]
        # .loc[{'Dp':slice(0,40e-9)}]
        .loc[{"hour": slice(0, 24)}]
        .coarsen(**{"hour": 5}, boundary="trim")
        .median()
        .where(lambda d: d["q"] > 95)["dndlDp"]
        .median("day")
        .plot(yscale="log", norm=mpl.colors.LogNorm())
    )


def plot_m(DS):
    (
        DS.loc[{"id": "nd"}]
        # .loc[{'Dp':slice(0,40e-9)}]
        .loc[{"hour": slice(0, 24)}]
        .coarsen(**{"hour": 5}, boundary="trim")
        .median()
        .where(lambda d: d["q"] > 95, drop=True)
        .assign_coords(
            {"N45": lambda d: d["dndlDp"].loc[{"Dp": slice(3.5e-9, 4.5e-9)}].mean("Dp")}
        )["dndlDp"]
        .pipe(lambda d: xr.corr(d, d["N45"], "day"))
        .plot(yscale="log", vmin=-1, vmax=1, cmap="RdBu_r")
    )


def plot_nn(DS):
    sciplot_def()
    (
        DS.loc[{"id": "nais"}]
        .loc[{"Dp": slice(0, 40e-9)}]
        .loc[{"hour": slice(0, 24)}]
        .coarsen(**{"hour": 5}, boundary="trim")
        .median()
        # .where(lambda d:d['q']>95,drop=True)
        .assign_coords(
            {"N45": lambda d: d["dndlDp"].loc[{"Dp": slice(3.5e-9, 4.5e-9)}].sum("Dp")}
        )["dndlDp"]
        .pipe(lambda d: xr.corr(d, d["N45"], ["day", "hour"]))
        .plot(xscale="log")
    )


def plot_below(MM, se):
    delta_n____ = "Ranking $\Delta N_{2.5-5}$"
    max_total_conc_ = f"Ranking {MM}\ntotal conc."
    f__pdf = f"f01_{MM}.pdf"
    sciplot_def(grid=False)
    sns.scatterplot(
        data=se["mTR"].to_dataframe(),
        y="mTR",
        x="q",
        s=5,
        alpha=0.3,
        hue="g",
        palette=["C1", "C2", "C4"],
        # scatter = False
    )
    # noinspection DuplicatedCode
    ax: plt.Axes = plt.gca()
    f: plt.Figure = plt.gcf()
    f.set_size_inches(3, 2)
    ax.set_box_aspect(1)
    set_margin(f, x1=0.7, y1=0.5, x2=1)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax = plt.gca()
    ax.get_legend().set_bbox_to_anchor([1, 1])
    ax.set_xlabel(delta_n____)
    ax.set_ylabel(max_total_conc_)
    _p_ = f__pdf
    plt.gcf().savefig(_p_, transparent=True)
    # region temp open fig
    os.system(f"open {_p_}")
    # endregion temp open fig

    # we are programming loike choppers. this should be programmed in a much better way.
    return se


def alternat_rank(d, MM):
    d1 = d["dndlDp"].sum("Dp")
    # d1.mean('hour')
    d1 = getattr(d1, MM)("hour")
    return d1


def get_se(ds, MM):

    se = (
        ds.where(lambda d: d["g"] != "", drop=True)
        .squeeze()
        .loc[{"hour": slice(0, 24)}]
        .loc[{"id": "nd"}]
        # below is for the ranking
        .rolling({"hour": 6}, center=True)
        .median()
        .assign({"n_max": lambda d: d["N_s"].max("hour")})  # .max('day')
        .assign(
            {"n_max_r": lambda d: d["n_max"].rank(pct=True, dim="day") * 100}
        )  # .max('day')
        .assign({"n_median": lambda d: d["N_s"].median("hour")})  # .median('day')
        .assign(
            {"n_median_r": lambda d: d["n_median"].rank(pct=True, dim="day") * 100}
        )  # .median('day')
        .assign({"n_mean": lambda d: d["N_s"].mean("hour")})  # .mean('day')
        .assign(
            {"n_mean_r": lambda d: d["n_mean"].rank(pct=True, dim="day") * 100}
        )  # .mean('day')
        # below we are smoothing
        .assign({"mT": lambda d: alternat_rank(d, MM)})  # .max('day')
        .assign(
            {"mTR": lambda d: d["mT"].rank(pct=True, dim="day") * 100}
        )  # .max('day')
        .assign({"dif": lambda d: (d["q"] - d["mTR"])})
        .drop("dndlDp")
        # .to_series()
    )
    return se


def bad_example_how_to_use_class_plot():

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
    ds,
    d1,
    new_name_class="Classification",
    new_name_perc_ranking="Percentile ranking",
    q_col_name="q",
):
    d5 = ds.pipe(lambda d: xr.merge([d, d1])).squeeze()
    npf_class = "npf_class"

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


def plot_class(mm, DS,aaa):
    nn = f"n_{mm}_r"
    # max_n____ = f"Ranking\n{mm} $ N_{{2.5-5}}$"
    # f__pdf = f"f02_{mm}.pdf"
    SE = get_se(DS, mm)
    clas_ds = import_class_ds()
    dsm = merge_ds_class(ds=SE, d1=clas_ds, new_name_perc_ranking=nn, q_col_name=nn)
    plot_hist(res=dsm, name_fig_out=nn + ".pdf", name_perc_ranking=nn, x_label=nn)

    get_50_quan(dsm, nn,aaa)

def get_50_quan_(dsm, nn,aaa):

    # np.interps = lambda x:x
    cla = "Classification"
    cl = "cl_1_0"
    r = (
        dsm[[nn, cla]]
        .dropna()
        .assign(
            **{cl: lambda d: d[cla].isin(["evetn II", "event Ib", "event Ia"])}
        )
        .sort_values(nn)
        # [[nn,cl]].cumsum()
        .assign(**{"cls": lambda d: d[cl].cumsum()})
        # .plot(x=nn,y=cl)
        .assign(**{'cls':lambda d: (d["cls"] / d["cls"].max())})
        .dropna()
        .pipe(lambda d: np.interp([.25,.5,.75],d['cls'],d[nn]))
    )
    print(r)
    aaa.append({nn:r})


def get_50_quan(dsm, nn,aaa):

    # np.interps = lambda x:x
    cla = "Classification"
    cl = "cl_1_0"
    r = (
        dsm[[nn, cla]]
        .dropna()
        .assign(
            **{cl: lambda d: d[cla].isin(["evetn II", "event Ib", "event Ia"])}
            )
        .sort_values(nn)
        # [[nn,cl]].cumsum()
        .assign(**{"cls": lambda d: d[cl].cumsum()})
        # .plot(x=nn,y=cl)
        .assign(**{'cls':lambda d: (d["cls"] / d["cls"].max())})
        .dropna()
        # .pipe(lambda d: np.interp([.25,.5,.75],d['cls'],d[nn]))
    )
    print(r)
    aaa.append({nn:r})


def plot_class_a(mm, DS,aaa):
    nn_ = 'mTR'
    nn = f"{nn_}_{mm}"
    # max_n____ = f"Ranking\n{mm} $ N_{{2.5-5}}$"
    # f__pdf = f"f02_{mm}.pdf"
    SE = get_se(DS, mm)
    clas_ds = import_class_ds()
    dsm = merge_ds_class(ds=SE, d1=clas_ds, new_name_perc_ranking=nn, q_col_name=nn_)
    plot_hist(res=dsm, name_fig_out=nn + ".pdf", name_perc_ranking=nn, x_label=nn)
    get_50_quan(dsm, nn,aaa)



def get_se_dn(ds, d1_, d2_ ):

    d1 = d1_ * 1e-9
    d2 = d2_ * 1e-9

    NN = f"DNN_R_{d1_}_{d2_}"

    se = (
        ds.where(lambda d: d["g"] != "", drop=True)
        .squeeze()
        .loc[{"hour": slice(-4, 24)}]
        .loc[{"id": "nd"}]
        # below is for the ranking
        .rolling({"hour": 7}, center=True)
        .median()
        .assign({f'NN':lambda d:d["dndlDp"].loc[{'Dp':slice(d1,d2)}].sum("Dp")})
        # below we are smoothing
        .assign({"NN_day_max": lambda d: d['NN'].where(d['is_day']).max('hour')})
        .assign({"NN_night_med": lambda d: d['NN'].where(d['is_night']).median('hour')})
        .assign({'DNN':lambda d:d['NN_day_max']-d['NN_night_med']})
        .assign(
            {NN: lambda d: d["DNN"].rank(pct=True, dim="day") * 100}
        )  # .max('day')
        # .assign({"dif": lambda d: (d["q"] - d["mTR"])})
        .drop("dndlDp")
        # .to_series()
    )
    return se, NN


def compare_class_d1_d2(DS,aaa, d1n, d2n):
    SE, nn = get_se_dn(DS, d1n, d2n)
    clas_ds = import_class_ds()
    dsm = merge_ds_class(ds=SE, d1=clas_ds, new_name_perc_ranking=nn, q_col_name=nn)
    plot_hist(res=dsm, name_fig_out=nn + ".pdf", name_perc_ranking=nn, x_label=nn)
    get_50_quan(dsm, nn, aaa)


def compare_with_orig_q(DS,aaa):
    mm = 'max' # i dont think it matters
    nn = f"q"
    # max_n____ = f"Ranking\n{mm} $ N_{{2.5-5}}$"
    # f__pdf = f"f02_{mm}.pdf"
    SE = get_se(DS, mm)
    clas_ds = import_class_ds()
    dsm = merge_ds_class(ds=SE, d1=clas_ds, new_name_perc_ranking=nn, q_col_name=nn)
    plot_hist(res=dsm, name_fig_out=nn + ".pdf", name_perc_ranking=nn, x_label=nn)
    get_50_quan(dsm, nn,aaa)

def plot_best_comp(aaa,dic_rep):
    sciplot_def(grid=False)
    dd = []
    for r in aaa:
        v = list(r.values())[0]
        print(v)
        d_ = {"nn": list(r.keys())[0], 25: v[0], 50: v[1], 75: v[2]}
        dd.append(d_)
    df = (
        pd.DataFrame(dd)
        # .replace(dic_rep)
        # .sort_values(25)[::-1]
        .reset_index(drop=True)
        .reset_index()
        .rename({"index": "c_index"}, axis=1)
        .set_index(["nn", "c_index"])
        .stack()
        .reset_index()
        .rename({0: "p"}, axis=1)
        .rename({"level_2": "pp"}, axis=1)
    )
    
    
    df1 = df.set_index(["nn", "c_index", "pp"])["p"].unstack().reset_index()

    # plt.rc({"ytick.family": "monospace"})
    # font = {'family':'monospace'}
    # plt.rc({'font':*font})
    # plt.rcParams["font.family"] = "monospace"


    sns.scatterplot(
        data=df,
        x="p",
        y="nn",
        hue="pp",
        legend=False,
        zorder=200,
        # style='pp',
        # markers = ['|','o','|']
    )
    
    
    ax: plt.Axes = plt.gca()
    ax.set_yticks([],minor=True)
    yt = ax.get_yticks()
    ax.set_yticks(yt,['']*len(yt))

    ax.hlines(y=df1["c_index"], xmin=df1[25], xmax=df1[75], colors=".75", zorder=100, lw=3)
    
    ax.set_xlim(None, 100)
    # ax.set_ylim(7,-1)
    ax.set_box_aspect(1.4)
    ax.grid(True, axis="y", c=".8")
    f: plt.Figure = plt.gcf()
    f.set_size_inches(2, 4)
    return df,df1


def from_df_to_xr(df,k):
    return (
        df[[k, "cls"]]
        .reset_index(drop=True)
        .rename({k: "val"}, axis=1)
        .to_xarray()
        .expand_dims({"pat": [k]})
    )

def plot_fig_d(aaa,dic_rep,
        fig_out = 'fig_cumsum.pdf'):



    vv = []
    kk = []
    for r in aaa:
        k = list(r.keys())[0]
        v = list(r.values())[0]
        vv.append(v)
        kk.append(k)
    dfs = []
    for i, (v_,k_) in enumerate(zip(vv[:],kk[:])):
        df_ = from_df_to_xr(v_,k_)
        dfs.append(df_)
    ds = xr.merge(dfs)

    ds2 = (
        ds
        .assign_coords({'i_val':lambda d:d['val'].median('pat')})
        .swap_dims({'index':'i_val'})
        .assign({'cls':lambda d:d['cls']*100})
        [['cls']]
        .assign({'cls_min':lambda d:d['cls'].min('pat')})
        .assign({'cls_dif':lambda d:d['cls']-d['cls_min']})
        .assign({'cls_dif_sum':lambda d:d['cls_dif'].mean('i_val')})
        # ['cls_dif_sum'].to_series().sort_values()
        # .pipe(lambda d:d.loc[{'pat':d[]}])
        .sortby('cls_dif_sum')
        # ['cls']
        # .to_dataframe()
        # ['']
        # .plot.scatter(x='i_val',y='cls',hue='pat')
    )
    l_va = len(ds2['pat'])
    # cmap = plt.get_cmap('plasma', l_va+2)
    cmap = plt.get_cmap('tab20',l_va)
    cmap = plt.get_cmap('tab20',l_va).colors[::]
    sciplot_def(grid=False)
    f, (ax,ax1) = plt.subplots(1,2,width_ratios=[3,2])


    def desp(ax1_):
        ax1_.spines['top'].set_visible(False)
        ax1_.spines['right'].set_visible(False)
        ax1_.spines['bottom'].set_visible(False)
        ax1_.spines['left'].set_visible(False)
        ax1_.tick_params(
            axis='both', which='both', bottom=False, top=False, left=False,
            right=False, labelbottom=False, labelleft=False
        )


    desp(ax1)

    f.set_size_inches(5,2.5)
    f:plt.Figure
    ax: plt.Axes

    from seaborn._oldcore import (
        unique_dashes,
        unique_markers,
    )
    


    values = ds2['pat'][::-1].to_series().values
    lv = len(values)

    um = unique_markers(lv)
    ud = unique_dashes(lv)

    ii = range(lv)[::]

    for i_,l in zip(ii,values):
        # print(l)
        z = 10
        lw = 1.5/2
        c_ = cmap[i_]
        if l =='q':
            z=100
            lw=2.5/2
            c_='k'
        # print(z)
        (
            ds2.loc[{'pat': l}]['cls']
            .plot(
                x='i_val', ax=ax, label=l,
                zorder=z,
                c=c_,
                lw=lw,
                # dashes = ud[i_]
                # marker=um[i_]
            )
        )
        # ax.set_ylim(.5,100)
        # ax.set_yscale('log')
        # ax.hlines([1.1+jj/4], [.5] ,[100.0],transform=ax.transAxes)
        ax1:plt.Axes
        # ax1.hlines([i_/12], [0] ,[.2], color=c_, lw=lw)
        ax1.scatter(.1,i_/12,marker=um[i_],c=[c_],zorder=10,ec='w',lw=.5)
        uu = [.5,1.2,2]
        tt = ['$D_1$[nm]','$D_2$[nm]','fun.']
        for jj in range(3):
            a_ = dic_rep[l][jj]
            ax1.text(uu[jj],i_/12,a_, va='center',ha='right',size=8)
            if i_==0:
                ax1.text(uu[jj],1.05,tt[jj], va='center',ha='right',size=10)

        ax1.set_xlim(0,2)
        ax1.set_ylim(-.05,1)


    ax.set_box_aspect(1/1.6)
    # ax.legend()
    # ax.get_legend().set_bbox_to_anchor([1,1])
    #region temp open fig
    ax.set_title('')
    ax.set_ylabel('cumulative events [\%]')
    ax.set_xlabel('ranking [\%]')
    # set_margin(f,x2=1.6,y1=.5)
    _p_ = fig_out;plt.gcf().savefig(_p_, transparent=True);
    os.system(f'open {_p_}')
    #endregion temp open fig
