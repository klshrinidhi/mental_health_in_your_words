import random
import sys
import pathlib
import pickle

from tqdm import tqdm,trange
import numpy as np
import polars as pl
import sklearn.metrics as skm
from autogluon.tabular import TabularDataset,TabularPredictor


def compute_metrics(labels_train, preds_train, labels_test, preds_test, anx_dep, n_samples, run, i_set, algo, is_multiclass):
    labels = [0,1,2,3] if is_multiclass else [0,1]
    train_labels,train_labels_counts = np.unique(labels_train,return_counts=True)
    ret = {f'train_example_count_{l}':c for l,c in zip(train_labels,train_labels_counts)}
    ret.update({'n_samples':n_samples,
                'run':run,
                'val_set':i_set,
                'algo':algo,
                'anx_dep':anx_dep,
                'val_acc':skm.accuracy_score(labels_test,preds_test),
                'val_balacc':skm.balanced_accuracy_score(labels_test,preds_test),
                'val_f1':skm.f1_score(labels_test,preds_test,average='weighted',labels=labels,zero_division=np.nan),
                'val_pr':skm.precision_score(labels_test,preds_test,average='weighted',labels=labels,zero_division=np.nan),
                'val_rc':skm.recall_score(labels_test,preds_test,average='weighted',labels=labels,zero_division=np.nan),
                'val_auc':skm.roc_auc_score(labels_test,preds_test,average='weighted',labels=labels),
                'train_acc':skm.accuracy_score(labels_train,preds_train),
                'train_balacc':skm.balanced_accuracy_score(labels_train,preds_train),
                'train_f1':skm.f1_score(labels_train,preds_train,average='weighted',labels=labels,zero_division=np.nan),
                'train_pr':skm.precision_score(labels_train,preds_train,average='weighted',labels=labels,zero_division=np.nan),
                'train_rc':skm.recall_score(labels_train,preds_train,average='weighted',labels=labels,zero_division=np.nan),
                'train_auc':skm.roc_auc_score(labels_train,preds_train,average='weighted',labels=labels)})
    return ret

def train(df_train, df_test, n_samples, run, i_set, anx_dep, out_f, is_multiclass):
    if out_f.is_file():
        metrics = pickle.load(out_f.open('rb'))
        return metrics
    metrics = list()
    df_train = TabularDataset(df_train)
    df_test = TabularDataset(df_test)
    out_d = out_f.parent / f'{n_samples:02}_{run:02}_{i_set}_{anx_dep}'
    if not out_d.is_dir():
        predictor = TabularPredictor(label=anx_dep,problem_type='binary',path=out_d,verbosity=0)
        predictor.fit(df_train,presets='best_quality')
    else:
        predictor = TabularPredictor.load(out_d)
    for model_name in tqdm(predictor.model_names(),desc=f'inference {anx_dep}',leave=False):
        pred_train = predictor.predict(data=df_train,model=model_name).to_numpy()
        pred_test = predictor.predict(data=df_test,model=model_name).to_numpy()
        Y_train = df_train[anx_dep].to_numpy()
        Y_test = df_test[anx_dep].to_numpy()
        metrics.append(compute_metrics(Y_train,pred_train,Y_test,pred_test,anx_dep,n_samples,run,i_set,model_name,is_multiclass))
    pickle.dump(metrics,out_f.open('wb'))
    return metrics

def train_ml_algos(val_participant_ids, df_sample, n_samples, i_run, out_d, is_multiclass=False, disable_tqdm=True):
    import warnings
    warnings.filterwarnings('ignore',category=FutureWarning)
    metrics_f = out_d / f'{n_samples:02}_{i_run:02}.pkl'
    if metrics_f.is_file():
        metrics = pickle.load(metrics_f.open('rb'))
        return metrics
    metrics = list()
    val_participant_ids = list(val_participant_ids)
    random.seed(100*n_samples+i_run)
    random.shuffle(val_participant_ids)
    for i_set in trange(5,desc='split',disable=disable_tqdm,leave=False):
        # Split the val_participant_ids into 5 parts, and use 4 for training and 1 for testing.
        val_pids = set(val_participant_ids[i_set::5])
        df_test = df_sample.filter(pl.col('val_participant_id').is_in(val_pids))
        df_train = df_sample.filter(pl.col('val_participant_id').is_in(val_pids).not_())
        df_test = df_test.pivot(on='train_participant_ids',index=['val_participant_id','anxiety','depression'],values=['anxiety_pd','depression_pd'])
        df_train = df_train.pivot(on='train_participant_ids',index=['val_participant_id','anxiety','depression'],values=['anxiety_pd','depression_pd'])
        df_test_anx = df_test.select(pl.exclude('val_participant_id','depression')).to_pandas()
        df_test_dep = df_test.select(pl.exclude('val_participant_id','anxiety')).to_pandas()
        df_train_anx = df_train.select(pl.exclude('val_participant_id','depression')).to_pandas()
        df_train_dep = df_train.select(pl.exclude('val_participant_id','anxiety')).to_pandas()
        anx_f = out_d / f'{n_samples:02}_{i_run:02}_{i_set}_anx.pkl'
        dep_f = out_d / f'{n_samples:02}_{i_run:02}_{i_set}_dep.pkl'
        try:
            m_anx = train(df_train_anx,df_test_anx,n_samples,i_run,i_set,'anxiety',anx_f,is_multiclass)
            m_dep = train(df_train_dep,df_test_dep,n_samples,i_run,i_set,'depression',dep_f,is_multiclass)
        except ValueError:
            # skm.roc_aud_score requires both classes in y_true or it will raise a ValueError with message
            # "Only one class present in y_true. ROC AUC score is not defined in that case."
            # Skip if this happens.
            continue
        metrics.extend(m_anx)
        metrics.extend(m_dep)
    pickle.dump(metrics,metrics_f.open('wb'))
    for f in list(out_d.glob(f'{n_samples:02}_{i_run:02}_*.pkl')):
        f.unlink(missing_ok=True)
    return metrics

if __name__ == '__main__':
    i_beg,i_end = int(sys.argv[1]),int(sys.argv[2])
    anxiety_labels = ['minimal','mild','moderate','severe']
    depression_labels = ['minimal','mild','moderate','moderately-severe','severe']
    mental_health_d = pathlib.Path('ROOT_D')
    # prompts_d = mental_health_d / 'prompts_1_11'
    # prompts_d = mental_health_d / 'prompts_1_12'
    # prompts_d = mental_health_d / 'prompts_1_13'
    prompts_d = mental_health_d / 'prompts_1_17'
    results_f = prompts_d / 'results.csv'
    # anxiety_pd_score_col = 'gpt-4o-2024-08-06\nanxiety total-score\nprompt_1_11'
    # anxiety_pd_score_col = 'gemini_1.5_pro_001\nanxiety total-score\nprompt_1_12'
    # anxiety_pd_score_col = 'llama_3.2_90b_vision_instruct\nanxiety total-score\nprompt_1_13'
    anxiety_pd_score_col = 'medlm_large_1.5\nanxiety total-score\nprompt_1_17'
    # depression_pd_score_col = 'gpt-4o-2024-08-06\ndepression total-score\nprompt_1_11'
    # depression_pd_score_col = 'gemini_1.5_pro_001\ndepression total-score\nprompt_1_12'
    # depression_pd_score_col = 'llama_3.2_90b_vision_instruct\ndepression total-score\nprompt_1_13'
    depression_pd_score_col = 'medlm_large_1.5\ndepression total-score\nprompt_1_17'
    # anxiety_pd_col = 'gpt-4o-2024-08-06\nanxiety\nprompt_1_11'
    # anxiety_pd_col = 'gemini_1.5_pro_001\nanxiety\nprompt_1_12'
    # anxiety_pd_col = 'llama_3.2_90b_vision_instruct\nanxiety\nprompt_1_13'
    anxiety_pd_col = 'medlm_large_1.5\nanxiety\nprompt_1_17'
    # depression_pd_col = 'gpt-4o-2024-08-06\ndepression\nprompt_1_11'
    # depression_pd_col = 'gemini_1.5_pro_001\ndepression\nprompt_1_12'
    # depression_pd_col = 'llama_3.2_90b_vision_instruct\ndepression\nprompt_1_13'
    depression_pd_col = 'medlm_large_1.5\ndepression\nprompt_1_17'
    df_results = pl.read_csv(results_f)[['val_participant_id','train_participant_ids',
                                         'anxiety','depression',anxiety_pd_col,depression_pd_col]]
    df_results_binary = (df_results
                         .with_columns(anxiety=pl.when(pl.col('anxiety') == 'minimal').then(0).otherwise(1),
                                       depression=pl.when(pl.col('depression') == 'minimal').then(0).otherwise(1),
                                       anxiety_pd=pl.col(anxiety_pd_col).replace_strict(['minimal','mild','moderate','severe'],[0,1,2,3]),
                                       depression_pd=pl.col(depression_pd_col).replace_strict(['minimal','mild','moderate','moderately-severe','severe'],[0,1,2,3,3]))
                         .drop(anxiety_pd_col,depression_pd_col))
    min_training_examples = 10
    ml_algos_d = prompts_d / 'ml_algos_binary'
    ml_algos_d.mkdir(parents=True,exist_ok=True)
    train_participant_ids = df_results_binary['train_participant_ids'].unique().sort()
    futures = list()
    for n_samples in trange(i_beg,i_end,1,desc='n_samples',leave=False):
        for i_run in trange(5,desc='run',leave=False):
            sample_train_participant_ids = train_participant_ids.sample(n_samples,seed=100*n_samples+i_run)
            df_sample = df_results_binary.filter(pl.col('train_participant_ids').is_in(sample_train_participant_ids))
            val_participant_ids = set(df_sample['val_participant_id'].unique().to_list())
            for pids in sample_train_participant_ids:
                vpids = df_sample.filter(pl.col('train_participant_ids') == pids)['val_participant_id'].unique().to_list()
                val_participant_ids.intersection_update(set(vpids))
            if len(val_participant_ids) < min_training_examples:
                continue
            df_sample = df_sample.filter(pl.col('val_participant_id').is_in(val_participant_ids))
            futures.append(train_ml_algos(val_participant_ids,df_sample,n_samples,i_run,ml_algos_d,False,False))
    metrics = [r for f in futures for r in f]
    pickle.dump(metrics,(ml_algos_d / 'metrics.pkl').open('wb'))
