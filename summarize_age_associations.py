# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:55:20 2025

@author: piercetf
"""

import polars as pl
from matplotlib import pyplot
import numpy as np
import seaborn as sb
from seaborn import objects as so

def create_labelcol(df, bf_col, name_col, count=20):
    
    thresh = df.select(pl.col(bf_col).top_k(count).min())[0,0]
    
    return (df
            .with_columns(
                pl.when(pl.col(bf_col).gt(thresh))
                .then(pl.col(name_col))
                .otherwise(pl.lit(''))
                .alias('label')
                )
            )

def draw_bf_volcano(df, log_bf, param, name='label', size=12):
    fig, ax = pyplot.subplots()
    fig.set_figheight(size)
    fig.set_figwidth(size)
    p = so.Plot(df, x=param, y=log_bf, text=name)
    p = p.add(so.Dots(marker='.'))
    p = p.add(so.Text(halign='left', valign='top', fontsize=20))
    p = p.on(ax)
    ax.axhline(0, linestyle=':')
    ax.axhline(0.5, linestyle='--')
    ax.axhline(1, linestyle='dashdot')
    ax.axhline(1.5, linestyle='-')
    ax.tick_params(labelsize=30)
    ax.set_xlabel(ax.get_xlabel(), fontsize=40)
    ax.set_ylabel(ax.get_ylabel(), fontsize=40)
    p.show()

FILE1 = "results/summary_simple.parquet"
FILE2 = "results/summary_cohortparams.parquet"

table1 = pl.read_parquet(FILE1)
table2 = pl.read_parquet(FILE2)

sub1 = (table1
        .select(
            pl.col('protein'),
            pl.col('base_cohort_sl_bf').log10().alias("log10(Bayes Factor)"),
            pl.col('base_cohort_summary').struct.field('mean').alias('hyperslope'),
            pl.col('base_cohort_summary').struct.field('std')
            )
        )

sub1 = create_labelcol(sub1, 'log10(Bayes Factor)', 'protein', 8)
draw_bf_volcano(sub1, 'log10(Bayes Factor)', 'hyperslope', 'label', 32)

sub2 = (table1
        .select(
            pl.col('protein'),
            pl.col('sex_slope_bf').log10().alias('log10(Bayes Factor)'),
            pl.col('sex_diff_summary').struct.field('mean').alias('sex diff (slope)'),
            pl.col('sex_diff_summary').struct.field('std')
            )
        )

sub2 = create_labelcol(sub2, 'log10(Bayes Factor)', 'protein', 10)
draw_bf_volcano(sub2, 'log10(Bayes Factor)', 'sex diff (slope)', 'label', 32)

sub3 = (table1
        .select(
            pl.col('protein'),
            pl.col("strain_slope_bf").log10().alias('log10(Bayes Factor)'),
            pl.col("strain_diff_summary").struct.field("mean").alias("strain diff (slope)"),
            pl.col("strain_diff_summary").struct.field("std")
            )
        )
sub3 = create_labelcol(sub3, 'log10(Bayes Factor)', 'protein', 10)
draw_bf_volcano(sub3, 'log10(Bayes Factor)', 'strain diff (slope)', 'label', 32)

combo = table1.join(table2,
                    on=pl.col('protein'),
                    how='left',
                    validate='1:m'
                    )
combo = combo.with_columns(
    pl.col('cohort_slope_bf_ndim').log10().alias('log10(Bayes Factor)'),
    pl.col('mean').alias('cohort slope')
    )


combo = create_labelcol(combo, 'log10(Bayes Factor)', 'protein')
facet = sb.relplot(combo, y='log10(Bayes Factor)', x='cohort slope', col='index', col_wrap=5)
for ax in facet.axes:
    ax.axhline(0, linestyle=':')
pyplot.show()





