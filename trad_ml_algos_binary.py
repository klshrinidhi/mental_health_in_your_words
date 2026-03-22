import pickle
import pathlib

import numpy as np
import sklearn.metrics as skm
import polars as pl
from tqdm.notebook import tqdm
import sklearn.ensemble as ske
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


mental_health_d = pathlib.Path('<MENTAL_HEALTH_D>')
results_f = mental_health_d / 'results_0shot.csv'
embeddings_f = mental_health_d / 'embeddings_openai_per_participant.pkl'
# embeddings_f = mental_health_d / 'embeddings_google_per_participant.pkl'
D = 256

###################################################################################################

df_results = (pl.read_csv(results_f)
              .sort('participant_id')
              ['participant_id','anxiety','depression'])
embeddings = pickle.load(open(embeddings_f,'rb'))

embs = np.array([embeddings[participant_id][D]
                 for participant_id in df_results['participant_id']])
emb_col_names = [f'emb{d:04}' for d in range(D)]
emb_cols = [pl.Series(f'emb{d:04}',embs[:,d])
            for d in range(D)]
df_results = df_results.with_columns(*emb_cols)

df_results = (df_results
              .with_columns(anxiety=pl.when(pl.col('anxiety') == 'minimal').then(0).otherwise(1),
                            depression=pl.when(pl.col('depression') == 'minimal').then(0).otherwise(1)))
labels = [0,1]

set_name,algo_name = list(),list()
anx_acc_train,dep_acc_train,anx_acc_test,dep_acc_test = list(),list(),list(),list()
anx_balacc_train,dep_balacc_train,anx_balacc_test,dep_balacc_test = list(),list(),list(),list()
anx_f1_train,dep_f1_train,anx_f1_test,dep_f1_test = list(),list(),list(),list()
anx_precision_train,dep_precision_train,anx_precision_test,dep_precision_test = list(),list(),list(),list()
anx_recall_train,dep_recall_train,anx_recall_test,dep_recall_test = list(),list(),list(),list()
anx_auc_train,dep_auc_train,anx_auc_test,dep_auc_test = list(),list(),list(),list()

def compute_metrics_anx(labels_train,preds_train,labels_test,preds_test):
    anx_acc_train.append(skm.accuracy_score(labels_train,preds_train))
    anx_acc_test.append(skm.accuracy_score(labels_test,preds_test))
    anx_balacc_train.append(skm.balanced_accuracy_score(labels_train,preds_train))
    anx_balacc_test.append(skm.balanced_accuracy_score(labels_test,preds_test))
    anx_f1_train.append(skm.f1_score(labels_train,preds_train,average='weighted',labels=labels))
    anx_f1_test.append(skm.f1_score(labels_test,preds_test,average='weighted',labels=labels))
    anx_precision_train.append(skm.precision_score(labels_train,preds_train,average='weighted',labels=labels))
    anx_precision_test.append(skm.precision_score(labels_test,preds_test,average='weighted',labels=labels))
    anx_recall_train.append(skm.recall_score(labels_train,preds_train,average='weighted',labels=labels))
    anx_recall_test.append(skm.recall_score(labels_test,preds_test,average='weighted',labels=labels))
    anx_auc_train.append(skm.roc_auc_score(labels_train,preds_train,average='weighted',labels=labels))
    anx_auc_test.append(skm.roc_auc_score(labels_test,preds_test,average='weighted',labels=labels))

def compute_metrics_dep(labels_train,preds_train,labels_test,preds_test):
    dep_acc_train.append(skm.accuracy_score(labels_train,preds_train))
    dep_acc_test.append(skm.accuracy_score(labels_test,preds_test))
    dep_balacc_train.append(skm.balanced_accuracy_score(labels_train,preds_train))
    dep_balacc_test.append(skm.balanced_accuracy_score(labels_test,preds_test))
    dep_f1_train.append(skm.f1_score(labels_train,preds_train,average='weighted',labels=labels))
    dep_f1_test.append(skm.f1_score(labels_test,preds_test,average='weighted',labels=labels))
    dep_precision_train.append(skm.precision_score(labels_train,preds_train,average='weighted',labels=labels))
    dep_precision_test.append(skm.precision_score(labels_test,preds_test,average='weighted',labels=labels))
    dep_recall_train.append(skm.recall_score(labels_train,preds_train,average='weighted',labels=labels))
    dep_recall_test.append(skm.recall_score(labels_test,preds_test,average='weighted',labels=labels))
    dep_auc_train.append(skm.roc_auc_score(labels_train,preds_train,average='weighted',labels=labels))
    dep_auc_test.append(skm.roc_auc_score(labels_test,preds_test,average='weighted',labels=labels))

pbar = tqdm(list(enumerate(sorted(mental_health_d.glob('val_*.csv')),1)),desc='val-set',leave=False)
for i,val_f in pbar:
    val_participant_ids = pl.read_csv(val_f)['participant_id']
    df_train = df_results.filter(pl.col('participant_id').is_in(val_participant_ids).not_())
    df_val = df_results.filter(pl.col('participant_id').is_in(val_participant_ids))
    X_train = df_train[emb_col_names].to_numpy()
    Y_train_anx = df_train['anxiety'].to_numpy()
    Y_train_dep = df_train['depression'].to_numpy()
    X_test = df_val[emb_col_names].to_numpy()
    Y_test_anx = df_val['anxiety'].to_numpy()
    Y_test_dep = df_val['depression'].to_numpy()
    pbar.set_postfix_str('KNeighborsClassifier anxiety')
    set_name.append(val_f.stem.replace('val_',''))
    algo_name.append('knn')
    nn_anx = KNeighborsClassifier()
    nn_anx.fit(X_train,Y_train_anx)
    compute_metrics_anx(Y_train_anx,nn_anx.predict(X_train),Y_test_anx,nn_anx.predict(X_test))
    pbar.set_postfix_str('KNeighborsClassifier depression')
    nn_dep = KNeighborsClassifier()
    nn_dep.fit(X_train,Y_train_dep)
    compute_metrics_dep(Y_train_dep,nn_dep.predict(X_train),Y_test_dep,nn_dep.predict(X_test))
    pbar.set_postfix_str('SVC Linear anxiety')
    set_name.append(val_f.stem.replace('val_',''))
    algo_name.append('svcl')
    sl_anx = SVC(kernel='linear')
    sl_anx.fit(X_train,Y_train_anx)
    compute_metrics_anx(Y_train_anx,sl_anx.predict(X_train),Y_test_anx,sl_anx.predict(X_test))
    pbar.set_postfix_str('SVC Linear depression')
    sl_dep = SVC(kernel='linear')
    sl_dep.fit(X_train,Y_train_dep)
    compute_metrics_dep(Y_train_dep,sl_dep.predict(X_train),Y_test_dep,sl_dep.predict(X_test))
    pbar.set_postfix_str('SVC RBF anxiety')
    set_name.append(val_f.stem.replace('val_',''))
    algo_name.append('svcr')
    sr_anx = SVC(kernel='rbf')
    sr_anx.fit(X_train,Y_train_anx)
    compute_metrics_anx(Y_train_anx,sr_anx.predict(X_train),Y_test_anx,sr_anx.predict(X_test))
    pbar.set_postfix_str('SVC RBF depression')
    sr_dep = SVC(kernel='rbf')
    sr_dep.fit(X_train,Y_train_dep)
    compute_metrics_dep(Y_train_dep,sr_dep.predict(X_train),Y_test_dep,sr_dep.predict(X_test))
    pbar.set_postfix_str('GaussianProcessClassifier anxiety')
    set_name.append(val_f.stem.replace('val_',''))
    algo_name.append('gp')
    gp_anx = GaussianProcessClassifier()
    gp_anx.fit(X_train,Y_train_anx)
    compute_metrics_anx(Y_train_anx,gp_anx.predict(X_train),Y_test_anx,gp_anx.predict(X_test))
    pbar.set_postfix_str('GaussianProcessClassifier depression')
    gp_dep = GaussianProcessClassifier()
    gp_dep.fit(X_train,Y_train_dep)
    compute_metrics_dep(Y_train_dep,gp_dep.predict(X_train),Y_test_dep,gp_dep.predict(X_test))
    pbar.set_postfix_str('DecisionTreeClassifier anxiety')
    set_name.append(val_f.stem.replace('val_',''))
    algo_name.append('dt')
    dt_anx = DecisionTreeClassifier()
    dt_anx.fit(X_train,Y_train_anx)
    compute_metrics_anx(Y_train_anx,dt_anx.predict(X_train),Y_test_anx,dt_anx.predict(X_test))
    pbar.set_postfix_str('DecisionTreeClassifier depression')
    dt_dep = DecisionTreeClassifier()
    dt_dep.fit(X_train,Y_train_dep)
    compute_metrics_dep(Y_train_dep,dt_dep.predict(X_train),Y_test_dep,dt_dep.predict(X_test))
    pbar.set_postfix_str('RandomForestClassifier anxiety')
    set_name.append(val_f.stem.replace('val_',''))
    algo_name.append('rf')
    rf_anx = ske.RandomForestClassifier()
    rf_anx.fit(X_train,Y_train_anx)
    compute_metrics_anx(Y_train_anx,rf_anx.predict(X_train),Y_test_anx,rf_anx.predict(X_test))
    pbar.set_postfix_str('RandomForestClassifier depression')
    rf_dep = ske.RandomForestClassifier()
    rf_dep.fit(X_train,Y_train_dep)
    compute_metrics_dep(Y_train_dep,rf_dep.predict(X_train),Y_test_dep,rf_dep.predict(X_test))
    pbar.set_postfix_str('MLPClassifier anxiety')
    set_name.append(val_f.stem.replace('val_',''))
    algo_name.append('mlp')
    mlp_anx = MLPClassifier(max_iter=1000)
    mlp_anx.fit(X_train,Y_train_anx)
    compute_metrics_anx(Y_train_anx,mlp_anx.predict(X_train),Y_test_anx,mlp_anx.predict(X_test))
    pbar.set_postfix_str('MLPClassifier depression')
    mlp_dep = MLPClassifier(max_iter=1000)
    mlp_dep.fit(X_train,Y_train_dep)
    compute_metrics_dep(Y_train_dep,mlp_dep.predict(X_train),Y_test_dep,mlp_dep.predict(X_test))
    pbar.set_postfix_str('AdaBoostClassifier anxiety')
    set_name.append(val_f.stem.replace('val_',''))
    algo_name.append('ab')
    ab_anx = ske.AdaBoostClassifier(algorithm='SAMME')
    ab_anx.fit(X_train,Y_train_anx)
    compute_metrics_anx(Y_train_anx,ab_anx.predict(X_train),Y_test_anx,ab_anx.predict(X_test))
    pbar.set_postfix_str('AdaBoostClassifier depression')
    ab_dep = ske.AdaBoostClassifier(algorithm='SAMME')
    ab_dep.fit(X_train,Y_train_dep)
    compute_metrics_dep(Y_train_dep,ab_dep.predict(X_train),Y_test_dep,ab_dep.predict(X_test))
    pbar.set_postfix_str('GaussianNB anxiety')
    set_name.append(val_f.stem.replace('val_',''))
    algo_name.append('nb')
    nb_anx = GaussianNB()
    nb_anx.fit(X_train,Y_train_anx)
    compute_metrics_anx(Y_train_anx,nb_anx.predict(X_train),Y_test_anx,nb_anx.predict(X_test))
    pbar.set_postfix_str('GaussianNB depression')
    nb_dep = GaussianNB()
    nb_dep.fit(X_train,Y_train_dep)
    compute_metrics_dep(Y_train_dep,nb_dep.predict(X_train),Y_test_dep,nb_dep.predict(X_test))
    pbar.set_postfix_str('GradientBoostingClassifier anxiety')
    set_name.append(val_f.stem.replace('val_',''))
    algo_name.append('gb')
    gb_anx = ske.GradientBoostingClassifier()
    gb_anx.fit(X_train,Y_train_anx)
    compute_metrics_anx(Y_train_anx,gb_anx.predict(X_train),Y_test_anx,gb_anx.predict(X_test))
    pbar.set_postfix_str('GradientBoostingClassifier depression')
    gb_dep = ske.GradientBoostingClassifier()
    gb_dep.fit(X_train,Y_train_dep)
    compute_metrics_dep(Y_train_dep,gb_dep.predict(X_train),Y_test_dep,gb_dep.predict(X_test))

df_scores_train = pl.DataFrame({'split':'train',
                                'train_val_set_name':set_name,
                                'algo_name':algo_name,
                                'anx_acc':anx_acc_train,
                                'dep_acc':dep_acc_train,
                                'anx_balacc':anx_balacc_train,
                                'dep_balacc':dep_balacc_train,
                                'anx_f1':anx_f1_train,
                                'dep_f1':dep_f1_train,
                                'anx_precision':anx_precision_train,
                                'dep_precision':dep_precision_train,
                                'anx_recall':anx_recall_train,
                                'dep_recall':dep_recall_train,
                                'anx_auc':anx_auc_train,
                                'dep_auc':dep_auc_train})
df_scores_test = pl.DataFrame({'split':'test',
                               'train_val_set_name':set_name,
                               'algo_name':algo_name,
                               'anx_acc':anx_acc_test,
                               'dep_acc':dep_acc_test,
                               'anx_balacc':anx_balacc_test,
                               'dep_balacc':dep_balacc_test,
                               'anx_f1':anx_f1_test,
                               'dep_f1':dep_f1_test,
                               'anx_precision':anx_precision_test,
                               'dep_precision':dep_precision_test,
                               'anx_recall':anx_recall_test,
                               'dep_recall':dep_recall_test,
                               'anx_auc':anx_auc_test,
                               'dep_auc':dep_auc_test})
df_scores = pl.concat([df_scores_train,df_scores_test])
df_scores = df_scores.group_by(['split','algo_name']).agg(pl.exclude('train_val_set_name').mean()).sort('algo_name','split')
df_scores.write_csv(mental_health_d / 'accuracy_openai_embs_sklearn_classifiers_valsets_mean_binary.csv')
# df_scores.write_csv(mental_health_d / 'accuracy_google_embs_sklearn_classifiers_valsets_mean_binary.csv')