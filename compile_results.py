from itertools import chain
import pathlib
import pickle

from rich import print
import numpy as np
from scipy.stats import wilcoxon
import polars as pl
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import seaborn as sns


sns.set_theme(style='white',
              font_scale=2,
              rc={'xtick.bottom':True,
                  'ytick.left':True})
plt_px = 1/plt.rcParams['figure.dpi']

mental_health_d = pathlib.Path('ROOT_D')
N_SAMPLES = 5
N_RUNS = 10
BASE_ALGO = 'NeuralNetFastAI'
anxiety_labels = ['minimal','mild','moderate','severe']
depression_labels = ['minimal','mild','moderate','severe']

###################################################################################################

def compute_wilcoxon_score(x):
    x = x.to_list()
    if isinstance(x[0],list):
        x = list(chain(*x))
    res = wilcoxon(x,alternative='greater')
    return [res.statistic,res.pvalue]


def compute_binary_mean_metrics_nosampling(df_results, roc_auc=True):
    label_to_id = {'minimal':0,
                   'mild':1,
                   'moderate':1,
                   'moderately-severe':1,
                   'severe':1}
    df_results_binary = df_results.with_columns(
        pl.col('anxiety').replace_strict(label_to_id),
        pl.col('depression').replace_strict(label_to_id),
        pl.col('anxiety_pd').replace_strict(label_to_id),
        pl.col('depression_pd').replace_strict(label_to_id)
    )
    anx_gt_pd = (df_results_binary['anxiety'],df_results_binary['anxiety_pd'])
    dep_gt_pd = (df_results_binary['depression'],df_results_binary['depression_pd'])
    kw_args1 = {'average':'weighted',
                'labels':[0,1]}
    kw_args2 = {'average':'weighted',
                'labels':[0,1],
                'zero_division':np.nan}
    df_scores = [{
        # 'anx_acc':skm.accuracy_score(*anx_gt_pd),
        # 'dep_acc':skm.accuracy_score(*dep_gt_pd),
        'anx_balacc':skm.balanced_accuracy_score(*anx_gt_pd),
        'dep_balacc':skm.balanced_accuracy_score(*dep_gt_pd),
        'anx_f1':skm.f1_score(*anx_gt_pd,**kw_args2),
        'dep_f1':skm.f1_score(*dep_gt_pd,**kw_args2),
        'anx_pr':skm.precision_score(*anx_gt_pd,**kw_args2),
        'dep_pr':skm.precision_score(*dep_gt_pd,**kw_args2),
        'anx_rc':skm.recall_score(*anx_gt_pd,**kw_args2),
        'dep_rc':skm.recall_score(*dep_gt_pd,**kw_args2),
        'anx_auc':skm.roc_auc_score(*anx_gt_pd,**kw_args1) if roc_auc else 0,
        'dep_auc':skm.roc_auc_score(*dep_gt_pd,**kw_args1) if roc_auc else 0
    }]
    df_scores = pl.DataFrame(df_scores)
    df_scores = df_scores.select((pl.col('*')*100).round(1))
    return df_scores


def compute_multiclass_mean_metrics_nosampling(df_results):
    label_to_id = {'minimal':0,
                   'mild':1,
                   'moderate':2,
                   'moderately-severe':3,
                   'severe':3}
    df_results_multiclass = df_results.with_columns(
        pl.col('anxiety').replace_strict(label_to_id),
        pl.col('depression').replace_strict(label_to_id),
        pl.col('anxiety_pd').replace_strict(label_to_id),
        pl.col('depression_pd').replace_strict(label_to_id)
    )
    anx_gt_pd = (df_results_multiclass['anxiety'],df_results_multiclass['anxiety_pd'])
    dep_gt_pd = (df_results_multiclass['depression'],df_results_multiclass['depression_pd'])
    kw_args = {'average':'weighted',
                'labels':[0,1,2,3],
                'zero_division':np.nan}
    df_scores = [{
        # 'anx_acc':skm.accuracy_score(*anx_gt_pd),
        # 'dep_acc':skm.accuracy_score(*dep_gt_pd),
        'anx_balacc':skm.balanced_accuracy_score(*anx_gt_pd),
        'dep_balacc':skm.balanced_accuracy_score(*dep_gt_pd),
        'anx_f1':skm.f1_score(*anx_gt_pd,**kw_args),
        'dep_f1':skm.f1_score(*dep_gt_pd,**kw_args),
        'anx_pr':skm.precision_score(*anx_gt_pd,**kw_args),
        'dep_pr':skm.precision_score(*dep_gt_pd,**kw_args),
        'anx_rc':skm.recall_score(*anx_gt_pd,**kw_args),
        'dep_rc':skm.recall_score(*dep_gt_pd,**kw_args),
    }]
    df_scores = pl.DataFrame(df_scores)
    df_scores = df_scores.select((pl.col('*')*100).round(1))
    return df_scores


def compute_binary_mean_metrics(df_results, roc_auc=True):
    label_to_id = {'minimal':0,
                   'mild':1,
                   'moderate':1,
                   'moderately-severe':1,
                   'severe':1}
    df_results_binary = df_results.with_columns(
        pl.col('anxiety').replace_strict(label_to_id),
        pl.col('depression').replace_strict(label_to_id),
        pl.col('anxiety_pd').replace_strict(label_to_id),
        pl.col('depression_pd').replace_strict(label_to_id)
    )
    df_scores = list()
    train_participant_ids = df_results_binary['train_participant_ids'].unique().sort()
    for n_samples in range(N_SAMPLES,N_SAMPLES+1):
        for i_run in range(N_RUNS):
            sample_train_participant_ids = train_participant_ids.sample(n_samples,seed=100*n_samples+i_run)
            df_val = df_results_binary.filter(pl.col('train_participant_ids').is_in(sample_train_participant_ids))
            df_mean = df_val.group_by('val_participant_id').agg(
                pl.col('anxiety').first(),
                pl.col('depression').first(),
                pl.col('anxiety_pd').mean().round().cast(pl.Int64),
                pl.col('depression_pd').mean().round().cast(pl.Int64)
            )
            anx_gt_pd = (df_mean['anxiety'],df_mean['anxiety_pd'])
            dep_gt_pd = (df_mean['depression'],df_mean['depression_pd'])
            kw_args1 = {'average':'weighted',
                        'labels':[0,1]}
            kw_args2 = {'average':'weighted',
                        'labels':[0,1],
                        'zero_division':np.nan}
            df_scores.append({
                'samples':n_samples,
                'run':i_run,
                # 'anx_acc':skm.accuracy_score(*anx_gt_pd),
                # 'dep_acc':skm.accuracy_score(*dep_gt_pd),
                'anx_balacc':skm.balanced_accuracy_score(*anx_gt_pd),
                'dep_balacc':skm.balanced_accuracy_score(*dep_gt_pd),
                'anx_f1':skm.f1_score(*anx_gt_pd,**kw_args2),
                'dep_f1':skm.f1_score(*dep_gt_pd,**kw_args2),
                'anx_pr':skm.precision_score(*anx_gt_pd,**kw_args2),
                'dep_pr':skm.precision_score(*dep_gt_pd,**kw_args2),
                'anx_rc':skm.recall_score(*anx_gt_pd,**kw_args2),
                'dep_rc':skm.recall_score(*dep_gt_pd,**kw_args2),
                'anx_auc':skm.roc_auc_score(*anx_gt_pd,**kw_args1) if roc_auc else 0,
                'dep_auc':skm.roc_auc_score(*dep_gt_pd,**kw_args1) if roc_auc else 0
            })
    df_scores = pl.DataFrame(df_scores)
    df_scores_mean = df_scores.group_by('samples').agg((pl.exclude('run').mean()*100).round(1)).sort('samples')
    df_scores_std = df_scores.group_by('samples').agg((pl.exclude('run').std()*100).round(1)).sort('samples')
    return df_scores_mean,df_scores_std


def compute_multiclass_mean_metrics(df_results, roc_auc=True):
    label_to_id = {'minimal':0,
                   'mild':1,
                   'moderate':2,
                   'moderately-severe':3,
                   'severe':3}
    df_results_multiclass = df_results.with_columns(
        pl.col('anxiety').replace_strict(label_to_id),
        pl.col('depression').replace_strict(label_to_id),
        pl.col('anxiety_pd').replace_strict(label_to_id),
        pl.col('depression_pd').replace_strict(label_to_id)
    )
    df_scores = list()
    train_participant_ids = df_results['train_participant_ids'].unique().sort()
    for n_samples in range(N_SAMPLES,N_SAMPLES+1):
        for i_run in range(N_RUNS):
            sample_train_participant_ids = train_participant_ids.sample(n_samples,seed=100*n_samples+i_run)
            df_val = df_results_multiclass.filter(pl.col('train_participant_ids').is_in(sample_train_participant_ids))
            df_mean = df_val.group_by('val_participant_id').agg(
                pl.col('anxiety').first(),
                pl.col('depression').first(),
                pl.col('anxiety_pd').mean().round().cast(pl.Int64),
                pl.col('depression_pd').mean().round().cast(pl.Int64)
            )
            anx_gt_pd = (df_mean['anxiety'],df_mean['anxiety_pd'])
            dep_gt_pd = (df_mean['depression'],df_mean['depression_pd'])
            anx_gt_pd_mat = (df_mean['anxiety'],np.eye(4)[df_mean['anxiety_pd']])
            dep_gt_pd_mat = (df_mean['depression'],np.eye(4)[df_mean['depression_pd']])
            kw_args1 = {'average':'weighted',
                        'labels':[0,1,2,3],
                        'multi_class':'ovr'}
            kw_args2 = {'average':'weighted',
                        'labels':[0,1,2,3],
                        'zero_division':np.nan}
            df_scores.append({
                'samples':n_samples,
                'run':i_run,
                # 'anx_acc':skm.accuracy_score(*anx_gt_pd),
                # 'dep_acc':skm.accuracy_score(*dep_gt_pd),
                'anx_balacc':skm.balanced_accuracy_score(*anx_gt_pd),
                'dep_balacc':skm.balanced_accuracy_score(*dep_gt_pd),
                'anx_f1':skm.f1_score(*anx_gt_pd,**kw_args2),
                'dep_f1':skm.f1_score(*dep_gt_pd,**kw_args2),
                'anx_pr':skm.precision_score(*anx_gt_pd,**kw_args2),
                'dep_pr':skm.precision_score(*dep_gt_pd,**kw_args2),
                'anx_rc':skm.recall_score(*anx_gt_pd,**kw_args2),
                'dep_rc':skm.recall_score(*dep_gt_pd,**kw_args2),
                'anx_auc':skm.roc_auc_score(*anx_gt_pd_mat,**kw_args1) if roc_auc else 0,
                'dep_auc':skm.roc_auc_score(*dep_gt_pd_mat,**kw_args1) if roc_auc else 0
            })
    df_scores = (pl.DataFrame(df_scores)
                 .with_columns(diff_anx_balacc=pl.col('anx_balacc')-0.25,
                               diff_dep_balacc=pl.col('dep_balacc')-0.25))
    df_scores_mean = (df_scores
                      .group_by('samples')
                      .agg((pl.exclude('run','diff_anx_balacc','diff_dep_balacc').mean()*100).round(1),
                           pl.col('diff_anx_balacc','diff_dep_balacc'))
                    #   .with_columns(wilcoxon_anx=pl.col('diff_anx_balacc').map_elements(compute_wilcoxon_score,
                    #                                                                     return_dtype=pl.List(pl.Float64)),
                    #                 wilcoxon_dep=pl.col('diff_dep_balacc').map_elements(compute_wilcoxon_score,
                    #                                                                     return_dtype=pl.List(pl.Float64)))
                      .drop('diff_anx_balacc','diff_dep_balacc')
                    #   .with_columns(wilcoxon_anx_stat=pl.col('wilcoxon_anx').list.get(0),
                    #                 wilcoxon_anx_pval=pl.col('wilcoxon_anx').list.get(1),
                    #                 wilcoxon_dep_stat=pl.col('wilcoxon_dep').list.get(0),
                    #                 wilcoxon_dep_pval=pl.col('wilcoxon_dep').list.get(1))
                    #   .with_columns(wilcoxon_anx_pval_str=pl.col('wilcoxon_anx_pval').map_elements(lambda x: f"{x:.1e}",return_dtype=pl.Utf8),
                    #                 wilcoxon_dep_pval_str=pl.col('wilcoxon_dep_pval').map_elements(lambda x: f"{x:.1e}",return_dtype=pl.Utf8))
                    #   .drop('wilcoxon_anx','wilcoxon_dep')
                      .sort('samples'))
    df_scores_std = (df_scores
                     .drop('diff_anx_balacc','diff_dep_balacc')
                     .group_by('samples')
                     .agg((pl.exclude('run').std()*100).round(1))
                     .sort('samples'))
    return df_scores_mean,df_scores_std


def compute_binary_majority_metrics(df_results):
    label_to_id = {'minimal':0,
                   'mild':1,
                   'moderate':1,
                   'moderately-severe':1,
                   'severe':1}
    df_results_binary = df_results.with_columns(
        pl.col('anxiety').replace_strict(label_to_id),
        pl.col('depression').replace_strict(label_to_id),
        pl.col('anxiety_pd').replace_strict(label_to_id),
        pl.col('depression_pd').replace_strict(label_to_id)
    )
    df_scores = list()
    train_participant_ids = df_results_binary['train_participant_ids'].unique().sort()
    for n_samples in range(N_SAMPLES,N_SAMPLES+1):
        for i_run in range(N_RUNS):
            sample_train_participant_ids = train_participant_ids.sample(n_samples,seed=100*n_samples+i_run)
            df_val = df_results_binary.filter(pl.col('train_participant_ids').is_in(sample_train_participant_ids))
            df_mode = df_val.group_by('val_participant_id').agg(
                pl.col('anxiety').first(),
                pl.col('depression').first(),
                pl.col('anxiety_pd').mode().first(),
                pl.col('depression_pd').mode().first()
            )
            anx_gt_pd = (df_mode['anxiety'],df_mode['anxiety_pd'])
            dep_gt_pd = (df_mode['depression'],df_mode['depression_pd'])
            kw_args1 = {'average':'weighted',
                        'labels':[0,1]}
            kw_args2 = {'average':'weighted',
                        'labels':[0,1],
                        'zero_division':np.nan}
            df_scores.append({
                'samples':n_samples,
                'run':i_run,
                # 'anx_acc':skm.accuracy_score(*anx_gt_pd),
                # 'dep_acc':skm.accuracy_score(*dep_gt_pd),
                'anx_balacc':skm.balanced_accuracy_score(*anx_gt_pd),
                'dep_balacc':skm.balanced_accuracy_score(*dep_gt_pd),
                'anx_f1':skm.f1_score(*anx_gt_pd,**kw_args2),
                'dep_f1':skm.f1_score(*dep_gt_pd,**kw_args2),
                'anx_pr':skm.precision_score(*anx_gt_pd,**kw_args2),
                'dep_pr':skm.precision_score(*dep_gt_pd,**kw_args2),
                'anx_rc':skm.recall_score(*anx_gt_pd,**kw_args2),
                'dep_rc':skm.recall_score(*dep_gt_pd,**kw_args2),
                'anx_auc':skm.roc_auc_score(*anx_gt_pd,**kw_args1),
                'dep_auc':skm.roc_auc_score(*dep_gt_pd,**kw_args1)
            })
    df_scores = pl.DataFrame(df_scores)
    df_scores_mean = df_scores.group_by('samples').agg((pl.exclude('run').mean()*100).round(1)).sort('samples')
    df_scores_std = df_scores.group_by('samples').agg((pl.exclude('run').std()*100).round(1)).sort('samples')
    return df_scores_mean,df_scores_std


def compute_multiclass_majority_metrics(df_results):
    label_to_id = {'minimal':0,
                   'mild':1,
                   'moderate':2,
                   'moderately-severe':3,
                   'severe':3}
    df_results_multiclass = df_results.with_columns(
        pl.col('anxiety').replace_strict(label_to_id),
        pl.col('depression').replace_strict(label_to_id),
        pl.col('anxiety_pd').replace_strict(label_to_id),
        pl.col('depression_pd').replace_strict(label_to_id)
    )
    df_scores = list()
    train_participant_ids = df_results['train_participant_ids'].unique().sort()
    for n_samples in range(N_SAMPLES,N_SAMPLES+1):
        for i_run in range(N_RUNS):
            sample_train_participant_ids = train_participant_ids.sample(n_samples,seed=100*n_samples+i_run)
            df_val = df_results_multiclass.filter(pl.col('train_participant_ids').is_in(sample_train_participant_ids))
            df_mode = df_val.group_by('val_participant_id').agg(
                pl.col('anxiety').first(),
                pl.col('depression').first(),
                pl.col('anxiety_pd').mode().first(),
                pl.col('depression_pd').mode().first()
            )
            anx_gt_pd = (df_mode['anxiety'],df_mode['anxiety_pd'])
            dep_gt_pd = (df_mode['depression'],df_mode['depression_pd'])
            kw_args = {'average':'weighted',
                       'labels':[0,1,2,3],
                       'zero_division':np.nan}
            df_scores.append({
                'samples':n_samples,
                'run':i_run,
                # 'anx_acc':skm.accuracy_score(*anx_gt_pd),
                # 'dep_acc':skm.accuracy_score(*dep_gt_pd),
                'anx_balacc':skm.balanced_accuracy_score(*anx_gt_pd),
                'dep_balacc':skm.balanced_accuracy_score(*dep_gt_pd),
                'anx_f1':skm.f1_score(*anx_gt_pd,**kw_args),
                'dep_f1':skm.f1_score(*dep_gt_pd,**kw_args),
                'anx_pr':skm.precision_score(*anx_gt_pd,**kw_args),
                'dep_pr':skm.precision_score(*dep_gt_pd,**kw_args),
                'anx_rc':skm.recall_score(*anx_gt_pd,**kw_args),
                'dep_rc':skm.recall_score(*dep_gt_pd,**kw_args)
            })
    df_scores = pl.DataFrame(df_scores)
    df_scores_mean = df_scores.group_by('samples').agg((pl.exclude('run').mean()*100).round(1)).sort('samples')
    df_scores_std = df_scores.group_by('samples').agg((pl.exclude('run').std()*100).round(1)).sort('samples')
    return df_scores_mean,df_scores_std


def save_binary_confusion_matrix(df_results, anx_out_f, dep_out_f):
    label_to_id = {'minimal':0,
                   'mild':1,
                   'moderate':1,
                   'moderately-severe':1,
                   'severe':1}
    df_results_binary = df_results.with_columns(
        pl.col('anxiety').replace_strict(label_to_id),
        pl.col('depression').replace_strict(label_to_id),
        pl.col('anxiety_pd').replace_strict(label_to_id),
        pl.col('depression_pd').replace_strict(label_to_id)
    )
    train_participant_ids = df_results_binary['train_participant_ids'].unique().sort()
    max_anx_balacc,max_dep_balacc = 0,0
    max_anx_cm,max_dep_cm = None,None
    for n_samples in range(N_SAMPLES,N_SAMPLES+1):
        for i_run in range(N_RUNS):
            sample_train_participant_ids = train_participant_ids.sample(n_samples,seed=100*n_samples+i_run)
            sample_train_participant_ids = set(sample_train_participant_ids.implode()[0])
            df_val = df_results_binary.filter(pl.col('train_participant_ids').is_in(sample_train_participant_ids))
            df_mean = df_val.group_by('val_participant_id').agg(
                pl.col('anxiety').first(),
                pl.col('depression').first(),
                pl.col('anxiety_pd').mean().round().cast(pl.Int64),
                pl.col('depression_pd').mean().round().cast(pl.Int64)
            )
            anx_gt_pd = (df_mean['anxiety'],df_mean['anxiety_pd'])
            dep_gt_pd = (df_mean['depression'],df_mean['depression_pd'])
            anx_balacc = skm.balanced_accuracy_score(*anx_gt_pd)
            dep_balacc = skm.balanced_accuracy_score(*dep_gt_pd)
            if anx_balacc > max_anx_balacc:
                max_anx_balacc = max(max_anx_balacc,anx_balacc)
                max_anx_cm = skm.confusion_matrix(*anx_gt_pd,labels=[0,1],normalize='true')
            if dep_balacc > max_dep_balacc:
                max_dep_balacc = max(max_dep_balacc,dep_balacc)
                max_dep_cm = skm.confusion_matrix(*dep_gt_pd,labels=[0,1],normalize='true')
    
    plt.figure(figsize=(500*plt_px,400*plt_px))
    sns.heatmap(max_anx_cm,annot=True,cmap='Blues',
                xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],
                vmin=0,vmax=1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Predicted',fontsize=28)
    plt.ylabel('Actual',fontsize=28)
    plt.tight_layout()
    plt.show()
    plt.savefig(anx_out_f,format='pdf',bbox_inches='tight')

    plt.figure(figsize=(500*plt_px,400*plt_px))
    sns.heatmap(max_dep_cm,annot=True,cmap='Blues',
                xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],
                vmin=0,vmax=1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Predicted',fontsize=28)
    plt.ylabel('Actual',fontsize=28)
    plt.tight_layout()
    plt.show()
    plt.savefig(dep_out_f,format='pdf',bbox_inches='tight')


def save_multiclass_confusion_matrix(df_results, anx_out_f, dep_out_f):
    label_to_id = {'minimal':0,
                   'mild':1,
                   'moderate':2,
                   'moderately-severe':3,
                   'severe':3}
    df_results_binary = df_results.with_columns(
        pl.col('anxiety').replace_strict(label_to_id),
        pl.col('depression').replace_strict(label_to_id),
        pl.col('anxiety_pd').replace_strict(label_to_id),
        pl.col('depression_pd').replace_strict(label_to_id)
    )
    train_participant_ids = df_results_binary['train_participant_ids'].unique().sort()
    max_anx_balacc,max_dep_balacc = 0,0
    max_anx_cm,max_dep_cm = None,None
    for n_samples in range(N_SAMPLES,N_SAMPLES+1):
        for i_run in range(N_RUNS):
            sample_train_participant_ids = train_participant_ids.sample(n_samples,seed=100*n_samples+i_run)
            sample_train_participant_ids = set(sample_train_participant_ids.implode()[0])
            df_val = df_results_binary.filter(pl.col('train_participant_ids').is_in(sample_train_participant_ids))
            df_mean = df_val.group_by('val_participant_id').agg(
                pl.col('anxiety').first(),
                pl.col('depression').first(),
                pl.col('anxiety_pd').mean().round().cast(pl.Int64),
                pl.col('depression_pd').mean().round().cast(pl.Int64)
            )
            anx_gt_pd = (df_mean['anxiety'],df_mean['anxiety_pd'])
            dep_gt_pd = (df_mean['depression'],df_mean['depression_pd'])
            anx_balacc = skm.balanced_accuracy_score(*anx_gt_pd)
            dep_balacc = skm.balanced_accuracy_score(*dep_gt_pd)
            if anx_balacc > max_anx_balacc:
                max_anx_balacc = max(max_anx_balacc,anx_balacc)
                max_anx_cm = skm.confusion_matrix(*anx_gt_pd,labels=[0,1,2,3],normalize='true')
            if dep_balacc > max_dep_balacc:
                max_dep_balacc = max(max_dep_balacc,dep_balacc)
                max_dep_cm = skm.confusion_matrix(*dep_gt_pd,labels=[0,1,2,3],normalize='true')

    plt.figure(figsize=(1000*plt_px,800*plt_px))
    sns.heatmap(max_anx_cm,annot=True,cmap='Blues',
                xticklabels=['Minimal','Mild','Moderate','Severe'],yticklabels=['Minimal','Mild','Moderate','Severe'],
                vmin=0,vmax=1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Predicted',fontsize=28)
    plt.ylabel('Actual',fontsize=28)
    plt.tight_layout()
    plt.show()
    plt.savefig(anx_out_f,format='pdf',bbox_inches='tight')

    plt.figure(figsize=(1000*plt_px,800*plt_px))
    sns.heatmap(max_dep_cm,annot=True,cmap='Blues',
                xticklabels=['Minimal','Mild','Moderate','Severe'],yticklabels=['Minimal','Mild','Moderate','Severe'],
                vmin=0,vmax=1)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Predicted',fontsize=28)
    plt.ylabel('Actual',fontsize=28)
    plt.tight_layout()
    plt.show()
    plt.savefig(dep_out_f,format='pdf',bbox_inches='tight')

    
def print_metrics(df_results):
    print('[bold blue]#####################################################################[/bold blue]')
    print('[bold blue]Binary Mean[/bold blue]')
    print('[bold blue]#####################################################################[/bold blue]')

    df_scores_mean,df_scores_std = compute_binary_mean_metrics(df_results)
    print(df_scores_mean)
    print(df_scores_std)

    print('[bold blue]#####################################################################[/bold blue]')
    print('[bold blue]Multiclass Mean[/bold blue]')
    print('[bold blue]#####################################################################[/bold blue]')

    df_scores_mean,df_scores_std = compute_multiclass_mean_metrics(df_results)
    print(df_scores_mean)
    print(df_scores_std)

    print('[bold blue]#####################################################################[/bold blue]')
    print('[bold blue]Binary Majority[/bold blue]')
    print('[bold blue]#####################################################################[/bold blue]')

    df_scores_mean,df_scores_std = compute_binary_majority_metrics(df_results)
    print(df_scores_mean)
    print(df_scores_std)

    print('[bold blue]#####################################################################[/bold blue]')
    print('[bold blue]Multiclass Majority[/bold blue]')
    print('[bold blue]#####################################################################[/bold blue]')

    df_scores_mean,df_scores_std = compute_multiclass_majority_metrics(df_results)
    print(df_scores_mean)
    print(df_scores_std)


def print_metrics_nosampling(df_results):
    print('[bold blue]#####################################################################[/bold blue]')
    print('[bold blue]Binary Mean[/bold blue]')
    print('[bold blue]#####################################################################[/bold blue]')
    df_scores = compute_binary_mean_metrics_nosampling(df_results)
    print(df_scores)

    print('[bold blue]#####################################################################[/bold blue]')
    print('[bold blue]Multiclass Mean[/bold blue]')
    print('[bold blue]#####################################################################[/bold blue]')
    df_scores = compute_multiclass_mean_metrics_nosampling(df_results)
    print(df_scores)


def load_scores(pkl_f):
    df = pickle.load(pkl_f.open('rb'))
    df = (
        pl.DataFrame(df)
        .select(pl.exclude('train_example_count_0','train_example_count_1',
                           'val_acc','train_acc')) # 'train_balacc','train_f1','train_pr','train_rc','train_auc'
        .with_columns(base_algo=pl.col('algo').str.split('_').list.get(0),
                      diff_balacc=pl.col('val_balacc')-0.5)
        .with_columns(pl.col('val_balacc')*100,pl.col('val_f1')*100,pl.col('val_pr')*100,pl.col('val_rc')*100,pl.col('val_auc')*100,
                      pl.col('train_balacc')*100,pl.col('train_f1')*100,pl.col('train_pr')*100,pl.col('train_rc')*100,pl.col('train_auc')*100)
    )
    df = (
        df
        .group_by('anx_dep','n_samples','run','base_algo','algo')
        .agg(pl.exclude('val_set','diff_balacc').mean(),pl.col('diff_balacc'))
    )
    df = (
        df
        .filter(pl.col('val_balacc') == pl.col('val_balacc').max().over('anx_dep','n_samples','run','base_algo'))
        .unique(subset=['anx_dep','n_samples','run','base_algo'],keep='first')
        .sort('n_samples','run','anx_dep','base_algo')
    )
    df_anx_mean = (
        df
        .filter(pl.col('anx_dep')=='anxiety')
        .group_by('n_samples','base_algo')
        .agg(pl.exclude('anx_dep','run','algo','diff_balacc').mean().round(1),pl.col('diff_balacc'))
        .sort('val_balacc',descending=True)
        .with_columns(wilcoxon=pl.col('diff_balacc').map_elements(compute_wilcoxon_score,
                                                                  return_dtype=pl.List(pl.Float64)))
        .with_columns(wilcoxon_stat=pl.col('wilcoxon').list.get(0),
                      wilcoxon_pval=pl.col('wilcoxon').list.get(1))
        .with_columns(wilcoxon_pval_str=pl.col('wilcoxon_pval').map_elements(lambda x: f"{x:.1e}",return_dtype=pl.Utf8))
        .drop('diff_balacc','wilcoxon')
    )
    df_dep_mean = (
        df
        .filter(pl.col('anx_dep')=='depression')
        .group_by('n_samples','base_algo')
        .agg(pl.exclude('anx_dep','run','algo','diff_balacc').mean().round(1),pl.col('diff_balacc'))
        .sort('val_balacc',descending=True)
        .with_columns(wilcoxon=pl.col('diff_balacc').map_elements(compute_wilcoxon_score,
                                                                  return_dtype=pl.List(pl.Float64)))
        .with_columns(wilcoxon_stat=pl.col('wilcoxon').list.get(0),
                      wilcoxon_pval=pl.col('wilcoxon').list.get(1))
        .with_columns(wilcoxon_pval_str=pl.col('wilcoxon_pval').map_elements(lambda x: f"{x:.1e}",return_dtype=pl.Utf8))
        .drop('diff_balacc','wilcoxon')
    )
    df_anx_std = (
        df
        .filter(pl.col('anx_dep')=='anxiety')
        .group_by('n_samples','base_algo')
        .agg(pl.exclude('anx_dep','run','algo','diff_balacc').std().round(1))
        .sort('val_balacc',descending=True)
    )
    df_dep_std = (
        df
        .filter(pl.col('anx_dep')=='depression')
        .group_by('n_samples','base_algo')
        .agg(pl.exclude('anx_dep','run','algo','diff_balacc').std().round(1))
        .sort('val_balacc',descending=True)
    )
    return df_anx_mean,df_dep_mean,df_anx_std,df_dep_std


###################################################################################################

print('[bold green]#####################################################################[/bold green]')
print('[bold green]ML Agg Algo[/bold green]')
print('[bold green]#####################################################################[/bold green]')

df_gpt_anx_mean,df_gpt_dep_mean,df_gpt_anx_std,df_gpt_dep_std = load_scores(mental_health_d / 'prompts_1_11' / 'ml_algos_binary' / 'metrics.pkl')
df_gem_anx_mean,df_gem_dep_mean,df_gem_anx_std,df_gem_dep_std = load_scores(mental_health_d / 'prompts_1_12' / 'ml_algos_binary' / 'metrics.pkl')
df_med_anx_mean,df_med_dep_mean,df_med_anx_std,df_med_dep_std = load_scores(mental_health_d / 'prompts_1_17' / 'ml_algos_binary' / 'metrics.pkl')
df_lla_anx_mean,df_lla_dep_mean,df_lla_anx_std,df_lla_dep_std = load_scores(mental_health_d / 'prompts_1_13' / 'ml_algos_binary' / 'metrics.pkl')

df_all_anx_mean = pl.concat([
    df_gpt_anx_mean.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO)),
    df_gem_anx_mean.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO)),
    df_med_anx_mean.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO)),
    df_lla_anx_mean.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO))
])
df_all_anx_mean = df_all_anx_mean.with_columns(pl.Series("model",["GPT-4o","Gemini-1.5pro","MedLM-Large-1.5","Llama-3.2-90b"]))
print(df_all_anx_mean)

df_all_anx_std = pl.concat([
    df_gpt_anx_std.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO)),
    df_gem_anx_std.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO)),
    df_med_anx_std.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO)),
    df_lla_anx_std.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO))
])
df_all_anx_std = df_all_anx_std.with_columns(pl.Series("model", ["GPT-4o", "Gemini-1.5pro", "MedLM-Large-1.5", "Llama-3.2-90b"]))
print(df_all_anx_std)

df_all_dep_mean = pl.concat([
    df_gpt_dep_mean.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO)),
    df_gem_dep_mean.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO)),
    df_med_dep_mean.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO)),
    df_lla_dep_mean.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO))
])
df_all_dep_mean = df_all_dep_mean.with_columns(pl.Series("model", ["GPT-4o", "Gemini-1.5pro", "MedLM-Large-1.5", "Llama-3.2-90b"]))
print(df_all_dep_mean)

df_all_dep_std = pl.concat([
    df_gpt_dep_std.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO)),
    df_gem_dep_std.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO)),
    df_med_dep_std.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO)),
    df_lla_dep_std.filter((pl.col('n_samples')==N_SAMPLES) & (pl.col('base_algo')==BASE_ALGO))
])
df_all_dep_std = df_all_dep_std.with_columns(pl.Series("model", ["GPT-4o", "Gemini-1.5pro", "MedLM-Large-1.5", "Llama-3.2-90b"]))
print(df_all_dep_std)

###################################################################################################

print('[bold green]#####################################################################[/bold green]')
print('[bold green]GPT-4o 1 shot[/bold green]')
print('[bold green]1 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_1_11'
results_f = prompts_d / 'results.csv'
anxiety_pd_score_col = 'gpt-4o-2024-08-06\nanxiety total-score\nprompt_1_11'
depression_pd_score_col = 'gpt-4o-2024-08-06\ndepression total-score\nprompt_1_11'
anxiety_pd_col = 'gpt-4o-2024-08-06\nanxiety\nprompt_1_11'
depression_pd_col = 'gpt-4o-2024-08-06\ndepression\nprompt_1_11'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)


print('[bold green]#####################################################################[/bold green]')
print('[bold green]GPT-4o 2 shot[/bold green]')
print('[bold green]2 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_2_05'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gpt-4o-2024-08-06\nanxiety\nprompt_2_05'
depression_pd_col = 'gpt-4o-2024-08-06\ndepression\nprompt_2_05'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)


print('[bold green]#####################################################################[/bold green]')
print('[bold green]GPT-4o 3 shot[/bold green]')
print('[bold green]3 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

results_f = mental_health_d / 'prompts_3_05' / 'results.csv'
anxiety_pd_col = 'gpt-4o-2024-08-06\nanxiety\nprompt_3_05'
depression_pd_col = 'gpt-4o-2024-08-06\ndepression\nprompt_3_05'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)


print('[bold green]#####################################################################[/bold green]')
print('[bold green]GPT-4o 4 shot[/bold green]')
print('[bold green]4 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_4_05'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gpt-4o-2024-08-06\nanxiety\nprompt_4_05'
depression_pd_col = 'gpt-4o-2024-08-06\ndepression\nprompt_4_05'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)


print('[bold green]#####################################################################[/bold green]')
print('[bold green]GPT-4o 5 shot[/bold green]')
print('[bold green]5 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_5_05'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gpt-4o-2024-08-06\nanxiety\nprompt_5_05'
depression_pd_col = 'gpt-4o-2024-08-06\ndepression\nprompt_5_05'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

###################################################################################################

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Gemini-1.5pro 1 shot[/bold green]')
print('[bold green]1 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_1_12'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gemini_1.5_pro_001\nanxiety\nprompt_1_12'
depression_pd_col = 'gemini_1.5_pro_001\ndepression\nprompt_1_12'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Gemini-1.5pro 2 shot[/bold green]')
print('[bold green]2 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_2_06'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gemini_1.5_pro_001\nanxiety\nprompt_2_06'
depression_pd_col = 'gemini_1.5_pro_001\ndepression\nprompt_2_06'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Gemini-1.5pro 3 shot[/bold green]')
print('[bold green]3 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]') 

prompts_d = mental_health_d / 'prompts_3_06'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gemini_1.5_pro_001\nanxiety\nprompt_3_06'
depression_pd_col = 'gemini_1.5_pro_001\ndepression\nprompt_3_06'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Gemini-1.5pro 4 shot[/bold green]')
print('[bold green]4 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_4_06'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gemini_1.5_pro_001\nanxiety\nprompt_4_06'
depression_pd_col = 'gemini_1.5_pro_001\ndepression\nprompt_4_06'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Gemini-1.5pro 5 shot[/bold green]')
print('[bold green]5 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_5_06'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gemini_1.5_pro_001\nanxiety\nprompt_5_06'
depression_pd_col = 'gemini_1.5_pro_001\ndepression\nprompt_5_06'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

###################################################################################################

print('[bold green]#####################################################################[/bold green]')
print('[bold green]MedLM-Large-1.5 1 shot[/bold green]')
print('[bold green]1 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_1_17'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'medlm_large_1.5\nanxiety\nprompt_1_17'
depression_pd_col = 'medlm_large_1.5\ndepression\nprompt_1_17'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]MedLM-Large-1.5 2 shot[/bold green]')
print('[bold green]2 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_2_11'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'medlm_large_1.5\nanxiety\nprompt_2_11'
depression_pd_col = 'medlm_large_1.5\ndepression\nprompt_2_11'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]MedLM-Large-1.5 3 shot[/bold green]')
print('[bold green]3 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_3_11'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'medlm_large_1.5\nanxiety\nprompt_3_11'
depression_pd_col = 'medlm_large_1.5\ndepression\nprompt_3_11'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]MedLM-Large-1.5 4 shot[/bold green]')
print('[bold green]4 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_4_11'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'medlm_large_1.5\nanxiety\nprompt_4_11'
depression_pd_col = 'medlm_large_1.5\ndepression\nprompt_4_11'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]MedLM-Large-1.5 5 shot[/bold green]')
print('[bold green]5 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_5_11'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'medlm_large_1.5\nanxiety\nprompt_5_11'
depression_pd_col = 'medlm_large_1.5\ndepression\nprompt_5_11'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

###################################################################################################

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Llama-3.2-90b 1 shot[/bold green]')
print('[bold green]1 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_1_13'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'llama_3.2_90b_vision_instruct\nanxiety\nprompt_1_13'
depression_pd_col = 'llama_3.2_90b_vision_instruct\ndepression\nprompt_1_13'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                             'anxiety','depression',
                                             anxiety_pd=anxiety_pd_col,
                                             depression_pd=depression_pd_col)
print_metrics(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Llama-3.2-90b 2 shot[/bold green]')
print('[bold green]2 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_2_07'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'llama_3.2_90b_vision_instruct\nanxiety\nprompt_2_07'
depression_pd_col = 'llama_3.2_90b_vision_instruct\ndepression\nprompt_2_07'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Llama-3.2-90b 3 shot[/bold green]')
print('[bold green]3 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_3_07'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'llama_3.2_90b_vision_instruct\nanxiety\nprompt_3_07'
depression_pd_col = 'llama_3.2_90b_vision_instruct\ndepression\nprompt_3_07'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Llama-3.2-90b 4 shot[/bold green]')
print('[bold green]4 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_4_07'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'llama_3.2_90b_vision_instruct\nanxiety\nprompt_4_07'
depression_pd_col = 'llama_3.2_90b_vision_instruct\ndepression\nprompt_4_07'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Llama-3.2-90b 5 shot[/bold green]')
print('[bold green]5 shot per class where the examples are randomly chosen[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_5_07'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'llama_3.2_90b_vision_instruct\nanxiety\nprompt_5_07'
depression_pd_col = 'llama_3.2_90b_vision_instruct\ndepression\nprompt_5_07'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics(df_results)

###################################################################################################

print('[bold green]#####################################################################[/bold green]')
print('[bold green]GPT-4o 1 shot[/bold green]')
print('[bold green]RAG 1 shot per class exp, where the 1 shot per class is [/bold green]')
print('[bold green]the nearest neighbor in the class as per cosine similarity on [/bold green]')
print('[bold green]OpenAI embeddings of D=3072.[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_1_09'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gpt-4o-2024-08-06\nanxiety\nprompt_1_09'
depression_pd_col = 'gpt-4o-2024-08-06\ndepression\nprompt_1_09'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics_nosampling(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Gemini-1.5pro 1 shot[/bold green]')
print('[bold green]RAG 1 shot per class exp, where the 1 shot is the nearest neighbor [/bold green]')
print('[bold green]per class as per cosine similarity on Google embeddings of D=768.[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_1_16'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gemini_1.5_pro_001\nanxiety\nprompt_1_16'
depression_pd_col = 'gemini_1.5_pro_001\ndepression\nprompt_1_16'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics_nosampling(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]GPT-4o 1 shot[/bold green]')
print('[bold green]RAG 1 shot exp, where the 1 shot is the nearest neighbor[/bold green]')
print('[bold green]in the whole dataset as per cosine similarity on[/bold green]')
print('[bold green]OpenAI embeddings of D=3072.[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompt_d = mental_health_d / 'prompts_1_10'
results_f = prompt_d / 'results.csv'
anxiety_pd_col = 'gpt-4o-2024-08-06\nanxiety\nprompt_1_10'
depression_pd_col = 'gpt-4o-2024-08-06\ndepression\nprompt_1_10'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics_nosampling(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Gemini-1.5pro 1 shot[/bold green]')
print('[bold green]RAG 1 shot exp, where the 1 shot is the nearest neighbor [/bold green]')
print('[bold green]in the whole dataset as per cosine similarity on [/bold green]')
print('[bold green]Google embeddings of D=768.[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_1_15'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gemini_1.5_pro_001\nanxiety\nprompt_1_15'
depression_pd_col = 'gemini_1.5_pro_001\ndepression\nprompt_1_15'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics_nosampling(df_results)

###################################################################################################

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Gemini-1.5pro 0 shot[/bold green]')
print('[bold green]Audio 0 shot[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_a_0_03'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gemini_1.5_pro_001\nanxiety\nprompt_a_0_03'
depression_pd_col = 'gemini_1.5_pro_001\ndepression\nprompt_a_0_03'
df_results = pl.read_csv(results_f).select('val_participant_id',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics_nosampling(df_results)

print('[bold green]#####################################################################[/bold green]')
print('[bold green]Gemini-1.5pro 1 shot[/bold green]')
print('[bold green]Audio 1 shot[/bold green]')
print('[bold green]#####################################################################[/bold green]')

prompts_d = mental_health_d / 'prompts_a_1_02'
results_f = prompts_d / 'results.csv'
anxiety_pd_col = 'gemini_1.5_pro_001\nanxiety\nprompt_a_1_02'
depression_pd_col = 'gemini_1.5_pro_001\ndepression\nprompt_a_1_02'
df_results = pl.read_csv(results_f).select('val_participant_id','train_participant_ids',
                                           'anxiety','depression',
                                           anxiety_pd=anxiety_pd_col,
                                           depression_pd=depression_pd_col)
print_metrics_nosampling(df_results)

###################################################################################################
