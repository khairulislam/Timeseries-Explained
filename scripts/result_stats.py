import pandas as pd
import numpy as np

def reduce_df(df:pd.DataFrame):
    # df[df['area']==0.05]
    return df.groupby('metric').aggregate('mean')[['comp', 'suff']].reset_index()

int_metric_map = {
    'electricity': ['mae'],
    'traffic': ['mae'],
    'mimic_iii': ['auc']
}

test_metric_map = {
    'electricity': ['mae', 'mse'],
    'traffic': ['mae', 'mse'],
    'mimic_iii': ['auc', 'accuracy', 'cross_entropy']
}

datasets = ['electricity', 'traffic', 'mimic_iii']
models = ['DLinear', 'MICN', 'SegRNN', 'iTransformer']
attr_methods = [
    'feature_ablation', 'augmented_occlusion', 
    'feature_permutation',
    'integrated_gradients', 'gradient_shap', 'dyna_mask',
    'winIT', 'tsr', 'gatemask', 'wtsr'
]

short_form = {
    'feature_ablation': 'FA',
    'occlusion':'FO',
    'augmented_occlusion': 'AFO',
    'feature_permutation': 'FP',
    'winIT': 'WinIT',
    'tsr':'TSR',
    'wtsr': 'WinTSR',
    'gradient_shap': 'GS',
    'integrated_gradients': 'IG',
    'dyna_mask': 'DM',
    'gatemask': 'ContraLSP'
}
NUM_ITERATIONS = 3

def print_row(item, decimals=2):
    if type(item) == str:
        print(f'& {item} ', end='')    
    else: print(f'& {np.round(item, decimals):03} ', end='')
    
def create_result_file(root='./results'):
    results = []
    for dataset in datasets:
        for attr_method in attr_methods:
            for metric in int_metric_map[dataset]:
                for model in models:
                    for itr_no in range(1, NUM_ITERATIONS+1):
                        df = pd.read_csv(f'{root}/{dataset}_{model}/{itr_no}/{attr_method}.csv')
                        # df = reduce_df(df)
                        values = df[df['metric']==metric][['area', 'comp', 'suff']].values
                        
                        for value in values:
                            area, comp, suff = value
                            results.append([
                                dataset, attr_method, metric, 
                                model, itr_no, area, comp, suff
                            ])

    result_df = pd.DataFrame(
        results, columns=['dataset', 'attr_method', 'metric', 
        'model', 'itr_no', 'area', 'comp', 'suff']
    )
    return result_df
    
print(f"Dataset & Metric &" + " & ".join(models) + " \\\\ \\hline")
for dataset in datasets:
    for metric in test_metric_map[dataset]:
        print(dataset, ' & ', metric, end='')
        for model in models:
            
            scores = 0
            for itr_no in range(1, NUM_ITERATIONS+1):
                df = pd.read_csv(f'results/{dataset}_{model}/{itr_no}/test_metrics.csv')
                score = df[df['metric']==metric]['score'].values[0]
                scores += score
                
            print_row(scores / NUM_ITERATIONS, decimals=3)
        print('\\\\')
        
# this section finds the ranks for each method 
result_df = create_result_file()
# result_df.round(3).to_csv('results/results.csv', index=False)

result_df = result_df.groupby(
    ['dataset', 'attr_method', 'metric', 'model', 'itr_no']
)[['comp', 'suff']].mean().reset_index()
print(result_df.head(3))

int_metrics = []
for dataset in int_metric_map:
    int_metrics.extend(int_metric_map[dataset])
int_metrics = list(set(int_metrics))

result_df = result_df[result_df['metric'].isin(int_metrics)]

selected = result_df['metric'].isin(['auc', 'accuracy'])
result_df.loc[selected, ['comp', 'suff']] = 1 - result_df[selected][['comp', 'suff']]

result_df['comp_rank'] = result_df.groupby(['dataset', 'metric', 'model'])['comp'].rank(ascending=False)
result_df['suff_rank'] = result_df.groupby(['dataset', 'metric', 'model'])['suff'].rank(ascending=True)
result_df.groupby(['dataset', 'metric', 'attr_method'])[['comp_rank', 'suff_rank']].mean().reset_index()

df = pd.concat([
    result_df.drop(columns='suff_rank').rename(columns={'comp_rank': 'rank'}), 
    result_df.drop(columns='comp_rank').rename(columns={'suff_rank': 'rank'})
], axis=0)

ranks = df.groupby(['dataset', 'metric', 'attr_method'])['rank'].mean().round(1).reset_index(name='mean_rank')
ranks['rank'] = ranks.groupby(['dataset', 'metric'])['mean_rank'].rank()
    
for dataset in datasets:
    # use the first or second on
    for metric in int_metric_map[dataset]:
        print(f'Dataset {dataset}, metric {metric}.\n')
        print(f" & {' & '.join(models)} & {' & '.join(models)} \\\\ \\hline")
        
        for attr_method in attr_methods:
            print(f'{short_form[attr_method]} ', end='')
            for metric_type in ['comp', 'suff']:
                for model in models:
                    scores = []
                    dfs = []
                    for itr_no in range(1, NUM_ITERATIONS+1):
                        df = pd.read_csv(f'results/{dataset}_{model}/{itr_no}/{attr_method}.csv') 
                    
                        df = df[df['metric']==metric][['area', metric_type]]
                        dfs.append(df)
                
                    df = pd.concat(dfs, axis=0)
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    if metric in ['auc', 'accuracy']:
                        df[metric_type] = 1-df[metric_type]
                    
                    score = df[metric_type].mean() 
                    print_row(score)
            
            mean_rank, rank = ranks[
                (ranks['dataset']==dataset) & (ranks['metric']==metric) & (ranks['attr_method']==attr_method)
            ][['mean_rank', 'rank']].values[0]
            
            print_row(f'{rank:.0f}({mean_rank})')
            print('\\\\')
        print('\\hline\n')

for dataset in datasets:
    # use the first or second on
    for metric in int_metric_map[dataset]:
        print(f'Dataset {dataset}, metric {metric}.\n')
        print(f" & {' & '.join(models)} \\\\ \\hline")
        
        for attr_method in attr_methods:
            print(f'{short_form[attr_method]} ', end='')
            for model in models:
                scores = []
                for metric_type in ['comp', 'suff']:
                    dfs = []
                    for itr_no in range(1, NUM_ITERATIONS+1):
                        df = pd.read_csv(f'results/{dataset}_{model}/{itr_no}/{attr_method}.csv') 
                    
                        df = df[df['metric']==metric][['area', metric_type]]
                        dfs.append(df)
                
                    df = pd.concat(dfs, axis=0)
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    if metric in ['auc', 'accuracy']:
                        df[metric_type] = 1-df[metric_type]
                    
                    score = df[metric_type].mean() 
                    scores.append(score)
            
                # take geometric mean of the two scores
                comp, suff = scores[0], scores[1]
                score = comp * (1-suff) / (comp + (1-suff))
                
                if dataset == 'mimic_iii': print_row(score, decimals=3)
                else: print_row(score, decimals=1)
            
            print('\\\\')
        print('\\hline\n')