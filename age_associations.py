# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:09:39 2025

@author: piercetf
"""

import polars as pl
from polars import selectors as cs
import pymc as pm
import arviz as az
from matplotlib import pyplot
import seaborn as sb
from sklearn import metrics
import numpy as np
import pathlib

def load_data():
    # first question: what the f*ck am I looking at
    PROT_TABLE = "./data/slam_900_counts_phenotypes.csv"
    EUTH_TABLE = "./data/Slam_900_phenotypes_combined w fast-fed.xlsx"
    #SMALL_TABLE = "./data/Slam_Serum_Proteins_30_sample_cohort.xlsx"
    
    p_tab = pl.read_csv(PROT_TABLE, null_values=["NA"])
    e_tab = pl.read_excel(EUTH_TABLE)
    #s_tab = pl.read_excel(SMALL_TABLE)
    
    # problem numero uno:
    # we need to combine euthanasia information and the protein information (small table doesn't match)
    # but the sample ID format is inconsistent regarding whether spaces are incluced
    # and the ID column has active discrepancies with it
    e_tab_fix = e_tab.select(
        pl.col("SLAMICS ID").str.replace(" ", "").alias("FIX SAMP ID"),
        pl.all()
    )
    
    p_tab_fix = p_tab.select(
        pl.col("SLAMICS ID").str.replace(" ", "").alias("FIX SAMP ID"),
        pl.all()
    )
    
    long_tab = e_tab_fix.join(
        p_tab_fix,
        on=pl.col("FIX SAMP ID"),
        how="left",
        validate="1:1")
    
    # problem numero dos:
    # euthanasia information is currently represented as a string
    # which is null when not euthanized and a text when it is
    # we want it as a boolean
    long_tab_e = long_tab.with_columns(
        is_euthanized = pl.col("Euthanasia Note").is_not_null()
    )
    
    # problem numero tres
    # sex and strain are string encoded
    # need numerical encoding
    # also going to numerize ID and Cohort
    # need separate id_rank column to identify mice later
    long_tab_e = long_tab_e.with_columns(
        ID_rank = pl.col("ID").rank("dense"),
    )
    long_tab_num = long_tab_e.to_dummies(["Sex", "Strain", "Cohort"])
    
    # problem numero cuatro
    # a major portion of proteins are observed so infrequently that its not clear whether they
    # are spurious identifications, rarely present, or rarely observed.
    # we will ignore if >5% are missingdata
    missingness = long_tab_num.null_count() / len(long_tab_num)
    tolerable_missing = (missingness.to_numpy() <= 0.10)[0]
    long_present = long_tab_num[:, tolerable_missing]
    
    # all remaining string-type columns are either duplicates or can be reconstructed from
    # numeric information using the `long_present` object.
    # they will now be removed to permit numerical treatment
    long_present_num_only = long_present.select(cs.numeric())
    # including factors presenting as numerics
    long_present_num_only = long_present_num_only.select(
        cs.exclude(cs.ends_with("_right"), # duplicates & irrelevant
                   pl.col("Visit") # don't care
                  )
    )
    
    return long_present_num_only


def process_protein(data, prot_name, output_dir):
    column_name = prot_name
    resfolderpath = output_dir
    if data[column_name].dtype != pl.Float64:
        return
    else:
        print(column_name)
    
    protfolderpath = resfolderpath / column_name
    
    try:
        protfolderpath.mkdir()
    except FileExistsError:
        # already exists
        pass
    
    N_MICE = data['ID_rank'].unique().shape[0]
    
    data = data.filter(pl.col(column_name).is_not_null())
    
    mean_p = data[column_name].mean()
    std_p = data[column_name].std()
    
    mean_age = data['Age'].mean()
    std_age = data['Age'].std()
    
    std_age = (data['Age'] - mean_age) / std_age
    std_prot = (data[column_name] - mean_p) / std_p
    
    with pm.Model() as model:
        mouse_id = pm.Data('mouse_id', data['ID_rank'] - 1)
        age = pm.Data('age', std_age)
        sex = pm.Data('sex', data['Sex_F'])
        strain = pm.Data('strain', data['Strain_B6'])
        
        base_intcpt = pm.Normal('base_intcpt', mu=0, sigma=1)
        mouse_intcpt = pm.Normal('mouse_intcpt', mu=base_intcpt, sigma=1, shape=N_MICE)
        
        slope = pm.Laplace('slope', mu=0, b=1)
        
        sex_slope = pm.Laplace('sex_slope', mu=0, b=1)
        
        strain_slope = pm.Laplace('strain_slope', mu=0, b=1)
        
        response = slope*age+mouse_intcpt[mouse_id]+sex_slope*sex*age + strain_slope*strain*age
        
        sigma = pm.Exponential('sigma', lam=10)
        
        _prot = pm.Normal(column_name, 
                         mu=response, 
                         sigma=sigma, 
                         observed=std_prot)
    
    with model:
        prior = pm.sample_prior_predictive()
        trace = pm.sample(nuts_sampler='blackjax',
                          tune=2000,
                          draws=5000,
                          random_seed=154112525)
        post = pm.sample_posterior_predictive(trace)
        trace.extend(prior)
        trace.extend(post)
        pm.compute_log_likelihood(trace, extend_inferencedata=True)
    
    bf_m = az.plot_bf(trace, 'slope', ref_val=0)
    if bf_m[0]['BF10'] > 10:
        pyplot.xlabel(f"slope, {column_name}, (male)")
        pyplot.savefig(protfolderpath / "slope_bf.svg", bbox_inches="tight")
        pyplot.show()
    else:
        pyplot.close()
        
    bf_f = az.plot_bf(trace, 'sex_slope', ref_val=0)
    if bf_f[0]['BF10'] > 10:
        pyplot.xlabel(f"sex_slope, {column_name}, (female)")
        pyplot.savefig(protfolderpath / "sex_slope_bf.svg", bbox_inches="tight")
        pyplot.show()
    else:
        pyplot.close()
    
    if bf_m[0]['BF10'] > 10 or bf_f[0]['BF10'] > 10:
        
        prot_pp = trace.posterior_predictive
        prot_modelpred = prot_pp.stack({"sample": ['chain', 'draw']})[column_name].transpose()
        act_prot = trace.observed_data[column_name]
        r2 = az.r2_score(act_prot, prot_modelpred)
        
        ped = metrics.PredictionErrorDisplay(
            y_true=np.array(trace.observed_data[column_name]),
            y_pred=np.array(prot_pp.mean(('chain','draw'))[column_name])
            )
        ped.plot(kind='actual_vs_predicted')
        pyplot.suptitle(f"{column_name} Actual vs Predicted (Std scale, in-sample)")
        pyplot.title(f"\nR² = {round(float(r2.r2),5)} +/- {round(float(r2.r2_std),5)}")
        pyplot.savefig(protfolderpath / "ped_act.svg", bbox_inches="tight")
        pyplot.show()
        
        ped.plot()
        pyplot.title(f"{column_name} residuals vs predictions")
        pyplot.savefig(protfolderpath / "ped_resid.svg", bbox_inches="tight")
        pyplot.show()
        
        az.plot_posterior(trace, ['slope', 'sex_slope', 'strain_slope'])
        pyplot.savefig(protfolderpath / "slope_posteriors.svg", bbox_inches="tight")
        pyplot.show()
        
        bf_strain = az.plot_bf(trace, 'strain_slope')
        pyplot.xlabel(f"slope diff for strains, {column_name}")
        pyplot.savefig(protfolderpath / "strain_slope_bf.svg", bbox_inches="tight")
        pyplot.show()
        
        #age_associated_proteins.append(column_name)
        breakpoint()
        return column_name, bf_m[0]['BF10'], bf_f[0]['BF10'], bf_strain[0]['BF10']
        
    else:
        bf_strain = az.plot_bf(trace, 'strain_slope')
        pyplot.show()
        print(f"evidence not meeting desired strength for {column_name}")
        protfolderpath.rmdir()
        return column_name, bf_m[0]['BF10'], bf_f[0]['BF10'], bf_strain[0]['BF10']
        

def main():
    this_script = pathlib.Path(__file__)
    resfolderpath = this_script.parent / 'results'
    try:
        resfolderpath.mkdir()
    except FileExistsError:
        # means it already exists, which is fine
        pass
    data = load_data()
    #age_associated_proteins = []
    table = {
        'gene': [],
        'slope_bf': [],
        'fem_slope_diff_bf': [],
        'B6_slope_diff_bf': [],
        }
    for column_name in data.columns:
        maybe_outcome = process_protein(data, column_name, resfolderpath)
        print(data.shape)
        if maybe_outcome is not None:
            # gene name and bayes factors for slope parameters
            gene, slope, f_slope, s_slope = maybe_outcome
            table['gene'].append(gene)
            table['slope_bf'].append(slope)
            table['fem_slope_diff_bf'].append(f_slope)
            table['B6_slope_diff_bf'].append(s_slope)
            
            pass
            #age_associated_proteins.append(maybe_colname)
    
    resframe = pl.from_dict(table)
    resframe.write_csv(resfolderpath / "slope_BFs.csv")
    return resframe


if __name__ == '__main__':
    outcomes = main()