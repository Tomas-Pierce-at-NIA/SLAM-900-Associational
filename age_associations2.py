# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 10:07:28 2025

@author: piercetf
"""

import polars as pl
from polars import selectors as cs
import pymc as pm
import arviz as az
from matplotlib import pyplot
import matplotlib as mpl
#import seaborn as sb
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
        Cohort_rank = pl.col('Cohort').rank('dense'),
    )
    long_tab_num = long_tab_e.to_dummies(["Sex", "Strain"])
    
    # problem numero cuatro
    # a major portion of proteins are observed so infrequently that its not clear whether they
    # are spurious identifications, rarely present, or rarely observed.
    # we will ignore if >10% are missingdata
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


class ProteinAnalysis:
    
    def __init__(self, prot_name :str, d_table :pl.DataFrame, folderpath :pathlib.Path):
        self.name = prot_name
        self.n_mice = d_table['ID_rank'].unique().shape[0]
        self.n_cohorts = d_table['Cohort_rank'].unique().shape[0]
        self.folder = folderpath / prot_name
        try:
            self.folder.mkdir()
        except FileExistsError:
            # file already exists, no work needed
            pass
        self.data = (d_table
                     .filter(
                         pl.col(prot_name).is_not_null()
                         )
                     .select(
                         pl.col('Age'),
                         pl.col('ID_rank'),
                         pl.col('Cohort_rank'),
                         pl.col('Sex_F'),
                         pl.col('Strain_B6'),
                         pl.col(prot_name)
                         )
                    )
        self.mean_age = self.data['Age'].mean()
        self.std_age = self.data['Age'].std()
        
        self.mean_p = self.data[prot_name].mean()
        self.std_p = self.data[prot_name].std()
        self.model = self.prior()
        
    
    def prior(self):
        
        std_age = (self.data['Age'] - self.mean_age) / self.std_age
        std_prot = (self.data[self.name] - self.mean_p) / self.std_p
        
        with pm.Model() as model:
            mouse_id = pm.Data('mouse_id', self.data['ID_rank'] - 1)
            cohort_id = pm.Data('cohort_id', self.data['Cohort_rank'] - 1)
            age = pm.Data('age', std_age)
            sex = pm.Data('sex', self.data['Sex_F'])
            strain = pm.Data('strain', self.data['Strain_B6'])
            
            base_intcpt = pm.Normal('base_intcpt', mu=0, sigma=1)
            mouse_intcpt = pm.Normal('mouse_intcpt', mu=base_intcpt, sigma=1, shape=self.n_mice)
            
            #slope = pm.Laplace('male_slope', mu=0, b=1)
            
            sex_slope_diff = pm.Laplace('sex_slope_diff', mu=0, b=1)
            
            strain_slope = pm.Laplace('strain_slope_diff', mu=0, b=1)
            
            #_female_slope = pm.Deterministic('female_slope', slope+sex_slope_diff)
            
            base_cohort_slope = pm.Laplace('base_cohort_slope', mu=0, b=1)
            cohort_slope = pm.Laplace('cohort_slope', mu=base_cohort_slope, b=1, shape=self.n_cohorts)
            
            response = (#slope*age + 
                        mouse_intcpt[mouse_id] +
                        sex_slope_diff*sex*age + 
                        strain_slope*strain*age +
                        cohort_slope[cohort_id] * age
                        )
            
            sigma = pm.Exponential('sigma', lam=10)
            
            # implicitly added to model
            # DO NOT REMOVE
            _prot = pm.Normal(self.name, 
                             mu=response, 
                             sigma=sigma, 
                             observed=std_prot)
        
        return model
    
    def fit(self):
        with self.model:
            prior = pm.sample_prior_predictive()
            trace = pm.sample(nuts_sampler='blackjax',
                              tune=2000,
                              draws=5000,
                              random_seed=154112525)
            post = pm.sample_posterior_predictive(trace)
            trace.extend(prior)
            trace.extend(post)
            pm.compute_log_likelihood(trace, extend_inferencedata=True)
        return trace
    
    def save_model_diagram(self):
        path = self.folder / "network_diag.svg"
        self.model.to_graphviz(save=str(path))
    
    def calc_r2(self, trace):
        pred = trace.posterior_predictive[self.name].stack(sample=('chain','draw')).T
        act = trace.observed_data[self.name]
        r2_obj = az.r2_score(y_true=act, y_pred=pred)
        r2 = R2Wrapper(r2_obj)
        return r2
    
    def __param_summary(self, trace, param_name):
        hdi = az.hdi(trace.posterior[param_name], 0.94)
        mean = trace.posterior[param_name].mean(('chain','draw'))
        std = trace.posterior[param_name].std(('chain','draw'))
        lower = hdi.sel(hdi='lower')[param_name].values
        higher = hdi.sel(hdi='higher')[param_name].values
        summary = {'mean': float(mean),
                   'std': float(std),
                   'hdi 94% low': float(lower),
                   'hdi 94% high': float(higher)}
        return summary
    
    def sex_diff_summary(self, trace):
        return self.__param_summary(trace, 'sex_slope_diff')
    
    def strain_diff_summary(self, trace):
        return self.__param_summary(trace, 'strain_slope_diff')
    
    def base_cohort_slope_summary(self, trace):
        return self.__param_summary(trace, 'base_cohort_slope')
    
    def cohort_slopes_summary(self, trace):
        return az.summary(trace, ['cohort_slope'])
    
    def save_slope_posterior(self, trace):
        az.plot_posterior(trace, ['base_cohort_slope', 'sex_slope_diff', 'strain_slope_diff', 'cohort_slope'])
        path = self.folder / "slope_posterior.svg"
        pyplot.savefig(path, bbox_inches="tight")
        pyplot.show()
    
    def __param_bf(self, trace, param_name):
        bfs = az.plot_bf(trace, param_name)
        bf_10 = bfs[0]['BF10']
        path = self.folder / f"{param_name}_bf.svg"
        pyplot.savefig(path, bbox_inches="tight")
        pyplot.show()
        return bf_10
    
    def base_cohort_slope_bf(self, trace):
        return self.__param_bf(trace, 'base_cohort_slope')
    
    def sex_diff_bf(self, trace):
        return self.__param_bf(trace, 'sex_slope_diff')
    
    def strain_diff_bf(self, trace):
        return self.__param_bf(trace, 'strain_slope_diff')
    
    def draw_ped(self, trace):
        r2 = self.calc_r2(trace)
        act = np.array(trace.observed_data[self.name])
        post_p = trace.posterior_predictive
        pred = np.array(post_p.mean(('chain', 'draw'))[self.name])
        ped = metrics.PredictionErrorDisplay(y_true=act, y_pred=pred)
        ped.plot(kind='actual_vs_predicted')
        path = self.folder / f"{self.name}_ped_actual.svg"
        pyplot.suptitle(f"{self.name} Actual vs Predicted (Std scale, in-sample)")
        pyplot.title(f"\nR² = {round(r2.r2,5)} +/- {round(r2.r2_std,5)}")
        pyplot.savefig(path)
        pyplot.show()
        
        ped.plot()
        path = self.folder / f"{self.name}_ped_residual.svg"
        pyplot.title(f"{self.name} residuals vs predictions")
        pyplot.savefig(path)
        pyplot.show
    
    def draw_model(self, trace):
        from matplotlib.lines import Line2D
        
        mouse_membership = (self.data
                            .select(
                                pl.col('ID_rank'),
                                pl.col('Cohort_rank'),
                                pl.col('Sex_F'),
                                pl.col('Strain_B6')
                                )
                            .unique()
                            )
        mouse_i = trace.posterior['mouse_intcpt'].mean(('chain','draw'))
        intercepts = mouse_i.sel(mouse_intcpt_dim_0=(mouse_membership['ID_rank']-1))
        cohort_sl = trace.posterior['cohort_slope'].mean(('chain', 'draw'))
        slope_part1 = cohort_sl.sel(cohort_slope_dim_0=(mouse_membership['Cohort_rank']-1))
        sex_slope = trace.posterior['sex_slope_diff'].mean(('chain', 'draw'))
        slope_part2 = mouse_membership['Sex_F'] * float(sex_slope)
        strain_slope = trace.posterior['strain_slope_diff'].mean(('chain', 'draw'))
        slope_part3 = mouse_membership['Strain_B6'] * float(strain_slope)
        slope = slope_part1.values + slope_part2 + slope_part3
        mouse_params = mouse_membership.with_columns(icpt_map = intercepts.values,
                                                     slopes = slope)
        
        intercept_bounds = az.hdi(trace.posterior['mouse_intcpt'], 0.94)
        cohort_sl_bounds = az.hdi(trace.posterior['cohort_slope'], 0.94)
        sex_sl_bounds = az.hdi(trace.posterior['sex_slope_diff'], 0.94)
        strain_sl_bounds = az.hdi(trace.posterior['strain_slope_diff'], 0.94)
        
        intercept_lowbound = intercept_bounds.sel(
            mouse_intcpt_dim_0=(mouse_membership['ID_rank']-1),
            hdi='lower'
            )
        intercept_upbound = intercept_bounds.sel(
            mouse_intcpt_dim_0=(mouse_membership['ID_rank']-1),
            hdi='higher'
            )
        
        cohort_lowbound = cohort_sl_bounds.sel(
            cohort_slope_dim_0=(mouse_membership['Cohort_rank']-1),
            hdi='lower',
            )
        cohort_upbound = cohort_sl_bounds.sel(
            cohort_slope_dim_0=(mouse_membership['Cohort_rank']-1),
            hdi='higher',
            )
        sex_lowbound = float(sex_sl_bounds.sel(hdi='lower')['sex_slope_diff'])
        sex_upbound = float(sex_sl_bounds.sel(hdi='higher')['sex_slope_diff'])
        
        strain_lowbound = float(strain_sl_bounds.sel(hdi='lower')['strain_slope_diff'])
        strain_upbound = float(strain_sl_bounds.sel(hdi='higher')['strain_slope_diff'])
        
        mouse_bounds = mouse_membership.with_columns(
            icpt_low = intercept_lowbound['mouse_intcpt'].values,
            icpt_high = intercept_upbound['mouse_intcpt'].values,
            cohort_low = cohort_lowbound['cohort_slope'].values,
            cohort_high = cohort_upbound['cohort_slope'].values,
            sex_low = pl.col('Sex_F') * sex_lowbound,
            sex_high = pl.col('Sex_F') * sex_upbound,
            strain_low = pl.col('Strain_B6') * strain_lowbound,
            strain_high = pl.col('Strain_B6') * strain_upbound
            )


        colors = mpl.color_sequences['tab10']
        
        fig, axs = pyplot.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
        fig.set_figwidth(60)
        fig.set_figheight(12)
        
        
        for i, ax in enumerate(axs.flat):
            cohort_rank = i + 1
            subtab = self.data.filter(pl.col('Cohort_rank').eq(cohort_rank))
            s_tab = subtab.with_columns(
                std_age = (pl.col('Age') - self.mean_age) / self.std_age,
                std_prot = (pl.col(self.name) - self.mean_p) / self.std_p,
                rank_id = pl.col('ID_rank').cast(pl.String),
                demograph = np.left_shift(pl.col('Sex_F'), 1) | pl.col('Strain_B6')
                )
            
            c_col = s_tab.select(
                pl.col('demograph').map_elements(lambda x : colors[x]).alias('color')
                )
            
            ax.scatter(x=s_tab['std_age'],
                       y=s_tab['std_prot'],
                       c=c_col['color'],
                       )
            
            leg_elem = [Line2D([0],[0],color=colors[0],marker='o',label='Male Het3'),
                     Line2D([0],[0],color=colors[1],marker='o',label='Male B6'),
                     Line2D([0],[0],color=colors[2],marker='o',label='Female Het3'),
                     Line2D([0],[0],color=colors[3],marker='o',label='Female B6')]
            
            ax.legend(handles=leg_elem)

            ax.set_title(f"Cohort (rank) {i + 1}")
            
            sub_params = mouse_params.filter(pl.col('Cohort_rank').eq(cohort_rank))
            #breakpoint()
            
            mean_params = (sub_params
                           .group_by(pl.col('Sex_F'), pl.col('Strain_B6'))
                           .mean()
                           .sort(pl.col('Sex_F'), pl.col('Strain_B6'))
                          )
            
            for i in range(mean_params.shape[0]):
                ax.axline((0, mean_params[i, 'icpt_map']),
                          slope=mean_params[i,'slopes'],
                          color=colors[i],
                          linestyle='-')
            
            sub_bounds = mouse_bounds.filter(pl.col('Cohort_rank').eq(cohort_rank))
            
            mean_bounds = (sub_bounds
                           .group_by(pl.col('Sex_F'), pl.col('Strain_B6'))
                           .mean()
                           .sort(pl.col('Sex_F'), pl.col('Strain_B6'))
                           )
            for i in range(mean_bounds.shape[0]):
                slope_low = (mean_bounds[i, 'cohort_low'] +
                             mean_bounds[i, 'sex_low'] + 
                             mean_bounds[i, 'strain_low']
                             )
                ax.axline((0, mean_bounds[i, 'icpt_low']),
                          slope=slope_low,
                          color=colors[i],
                          linestyle=':')
                
                slope_high = (mean_bounds[i, 'cohort_high'] +
                              mean_bounds[i, 'sex_high'] + 
                              mean_bounds[i, 'strain_high']
                              )
                ax.axline((0, mean_bounds[i, 'icpt_high']),
                          slope=slope_high,
                          color=colors[i],
                          linestyle=':')
                
                low_start = slope_low * s_tab['std_age'].min() + mean_bounds[i, 'icpt_low']
                low_end = slope_low * s_tab['std_age'].max() + mean_bounds[i, 'icpt_low']
                high_start = slope_high * s_tab['std_age'].min() + mean_bounds[i, 'icpt_high']
                high_end = slope_high * s_tab['std_age'].max() + mean_bounds[i, 'icpt_high']
                
                ax.fill_between(x=[s_tab['std_age'].min(), s_tab['std_age'].max()],
                                y1=[low_start, low_end],
                                y2=[high_start, high_end],
                                color=colors[i],
                                alpha=0.3
                                )

            
            ax.set_xlabel('Age (standard scale)')
            ax.set_ylabel(f"{self.name} (standard scale)")

            
        fig.suptitle(f"{self.name} data & MP prediction by subgroup")
        fig.tight_layout()
        
        path = self.folder / f"{self.name}_show_model.svg"
        pyplot.savefig(path, bbox_inches="tight")
        
        pyplot.show()
    
    def save_loo_pit(self, trace):
        az.plot_loo_pit(trace, self.name)
        path = self.folder / "loo_pit.svg"
        pyplot.savefig(path, bbox_inches="tight")
        pyplot.show()
    
    def save_cohort_bf(self, trace):
        bf = az.plot_bf(trace, 'cohort_slope')
        path = self.folder / "cohort_slope_bf.svg"
        pyplot.savefig(path, bbox_inches="tight")
        pyplot.show()
        return bf[0]['BF10']


class R2Wrapper:
    def __init__(self, r2_obj):
        self.r2 = float(r2_obj.r2)
        self.r2_std = float(r2_obj.r2_std)



def main():
    this_script = pathlib.Path(__file__)
    resfolderpath = this_script.parent / 'results'
    try:
        resfolderpath.mkdir()
    except FileExistsError:
        # folder already exists, no work needed
        pass
    
    results = {}
    data = load_data()
    for column_name in data.columns:
        if data[column_name].dtype != pl.Float64:
            continue
        plist = results.setdefault('protein', [])
        plist.append(column_name)
        pa = ProteinAnalysis(column_name, data, resfolderpath)
        pa.save_model_diagram()
        trace = pa.fit()
        pa.draw_model(trace)
        pa.draw_ped(trace)
        r2 = pa.calc_r2(trace)
        results.setdefault('r2', []).append(r2.r2)
        results.setdefault('r2_std', []).append(r2.r2_std)
        pa.save_slope_posterior(trace)
        base_cohort_bf = pa.base_cohort_slope_bf(trace)
        results.setdefault('base_cohort_sl_bf', []).append(base_cohort_bf)
        sex_slope_bf = pa.sex_diff_bf(trace)
        results.setdefault('sex_slope_bf', []).append(sex_slope_bf)
        strain_slope_bf = pa.strain_diff_bf(trace)
        results.setdefault('strain_slope_bf', []).append(strain_slope_bf)
        base_cohort_sl_sum = pa.base_cohort_slope_summary(trace)
        results.setdefault('base_cohort_summary', []).append(base_cohort_sl_sum)
        sex_sl_sum = pa.sex_diff_summary(trace)
        results.setdefault('sex_diff_summary', []).append(sex_sl_sum)
        strain_sl_sum = pa.strain_diff_summary(trace)
        results.setdefault('strain_diff_summary', []).append(strain_sl_sum)
        cohort_sl_summaries = pa.cohort_slopes_summary(trace)
        results.setdefault('cohort_sl_summary', []).append(cohort_sl_summaries)
        pa.save_loo_pit(trace)
        cohort_sl_bf = pa.save_cohort_bf(trace)
        results.setdefault('cohort_slope_bf_ndim', []).append(cohort_sl_bf)
    
    return results


if __name__ == '__main__':
    res_dict = main()
    res_table = pl.from_dict(res_dict)
    res_table_simple = res_table.select(pl.exclude('cohort_sl_summary'))
    res_table_simple.write_parquet('results/summary_simple.parquet')
    cohortsum = res_table.select(pl.col('protein'), pl.col('cohort_sl_summary'))
    subtables = []
    for p_name, subtable in cohortsum.rows():
        subtable['protein'] = p_name
        s_table = subtable.reset_index()
        subtables.append(pl.from_pandas(s_table))
    cohort_sums = pl.concat(subtables)
    cohort_sums.write_parquet("results/summary_cohortparams.parquet")