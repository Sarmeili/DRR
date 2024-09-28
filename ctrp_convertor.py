import pandas as pd
import numpy as np 
from keras.preprocessing.sequence import pad_sequences
import re

def ctrp_to_gdsc_format():

    def get_selected_genes():
        selected_genes = pd.read_csv('data/ctrp_data/selected_genes.csv', index_col=0)
        selected_genes = selected_genes['genes'].unique()
        pd.Series(selected_genes).to_csv('data/ctrp_processed_data/dicts_from_csvs/paccmann_gene_list_dict.csv', 
                                              index=False, header=False)
        return selected_genes
        
    def gene_expression():
        gene_exp = pd.read_csv('data/ctrp_data/CCLE_expression.csv')
        meta_ccle = pd.read_csv('data/ctrp_data/sample_info.csv')
        meta_ccle = meta_ccle[['DepMap_ID', 'stripped_cell_line_name']]
        gene_exp = gene_exp.merge(meta_ccle, left_on='Unnamed: 0', right_on='DepMap_ID', how='left')
        gene_exp.drop(['Unnamed: 0', 'DepMap_ID'], inplace=True, axis=1)
        gene_exp.set_index('stripped_cell_line_name', inplace=True)
        gene_exp.columns = [col.split(' ')[0] for col in gene_exp.columns]
        gene_exp = gene_exp.reset_index()
        gene_exp = gene_exp.rename(columns={'stripped_cell_line_name':'cell_line'})
        gene_exp.set_index('cell_line', inplace=True)
        selected_genes = get_selected_genes()
        selected_genes = [gene for gene in selected_genes if gene in gene_exp.columns]
        gene_exp = gene_exp[list(selected_genes)]
        gene_exp.to_csv('data/ctrp_processed_data/dicts_from_csvs/cell_line.csv')
        return gene_exp

    def drug():
        cpd_df = pd.read_csv('data/ctrp_data/v20.meta.per_compound.txt', sep='\t')
        cpd_df = cpd_df[['master_cpd_id', 'cpd_name', 'cpd_smiles']]
        drug_data = cpd_df[['cpd_name', 'cpd_name', 'cpd_smiles']]
        drug_data.columns = ['drug', 'drug_name', 'canonical_smiles']
        drug_data = drug_data.set_index('drug')
        drug_data.to_csv('data/ctrp_processed_data/dicts_from_csvs/drug_data.tsv')
        return drug_data
    
    def tokenize_smiles(smiles_list):

        token_id = pd.read_csv('data/gdsc_data/dicts_from_csvs/token_id_mapping.txt', 
                               sep='\t', names=['token', 'id'], header=None)
        token_id = dict(zip(token_id['token'], token_id['id']))
        token_id['@'] = 31
        token_id['@@'] = 32
        token_id['-'] = 34
        token_id['.'] = 33
        token_id['o'] = 3
        token_id['c'] = 9
        token_id['n'] = 10
        token_id['s'] = 23
        token_id["\\"] = 35
        token_id['[nH]'] = 28
        token_id['[C@@H]'] = 36
        token_id['[C@@]'] = 37
        token_id['[C@H]'] = 38
        token_id['[C@]'] = 39
        token_id['/'] = 40
        token_id['[Pt]'] = 41
        token_id['[As]'] = 42
        token_id['[n+]'] = 43
        token_id['[S@@]'] = 44
        token_id['[N@@]'] =45
        token_id['z'] = 46

        token_id_df = pd.DataFrame({'token' : token_id.keys(),
                           'id' : token_id.values()}).set_index('token')
        token_id_df.to_csv('data/ctrp_processed_data/dicts_from_csvs/token_id_mapping.txt', header=None, sep='\t')
    

        smiles_tokenized_list = []
        token_pattern = r'(\[.*?\]|Cl|Br|@@|@|.)'
        max_len_tokens = 0
        for smiles in smiles_list:
            tokens = re.findall(token_pattern, smiles)
            tokenized = [token_id.get(char, 0) for char in tokens]
            if len(tokenized)>=max_len_tokens:
                max_len_tokens = len(tokenized)
            smiles_tokenized_list.append(tokenized)
        
        padded_tokenized = pad_sequences(smiles_tokenized_list, maxlen=max_len_tokens, padding='pre', value=0)
        
        return padded_tokenized, max_len_tokens
    
    def five_folding_data():
        gene_exp = gene_expression()
        response_df = pd.read_csv('data/ctrp_data/v20.data.curves_post_qc.txt', sep='\t')
        response_df = response_df[['master_cpd_id', 'experiment_id', 'area_under_curve']]

        experiment_df = pd.read_csv('data/ctrp_data/v20.meta.per_experiment.txt', sep='\t')
        cll_df = pd.read_csv('data/ctrp_data/v20.meta.per_cell_line.txt', sep='\t')
        experiment_df = experiment_df[['experiment_id', 'master_ccl_id']]
        experiment_df.drop_duplicates(inplace=True)
        cll_df = cll_df[['master_ccl_id', 'ccl_name', 'ccle_primary_site', 'ccle_primary_hist']]

        cpd_df = pd.read_csv('data/ctrp_data/v20.meta.per_compound.txt', sep='\t')
        cpd_df = cpd_df[['master_cpd_id', 'cpd_name', 'cpd_smiles', 'cpd_status']]

        drug_txt = cpd_df[['cpd_name', 'cpd_status']]
        drug_txt.to_csv('data/ctrp_processed_data/drugs.txt', header=None, sep='\t')

        response_df = response_df.merge(cpd_df, on='master_cpd_id', how='left')
        response_df = response_df.merge(experiment_df, on='experiment_id', how='left')
        response_df = response_df.merge(cll_df, on='master_ccl_id', how='left')

        response_df = response_df[response_df['ccl_name'].isin(gene_exp.index)]

        clls_list = response_df['ccl_name'].unique()
        drugs_list = response_df['cpd_name'].unique()
        np.random.seed(12345)
        np.random.shuffle(clls_list)
        np.random.shuffle(drugs_list)
        
        for i in range(5):
            fold_size = int(len(clls_list)/5)
            if i == 4:
                test_clls = clls_list[i * fold_size : ]
            else:
                test_clls = clls_list[i * fold_size : (i + 1) * fold_size]
            
            train_clls = np.setdiff1d(clls_list, test_clls)

            test_df = response_df[response_df['ccl_name'].isin(test_clls)]
            test_pivot = test_df.pivot_table(index='ccl_name', columns='cpd_name', values='area_under_curve', aggfunc='mean')
            test_pivot = test_pivot.reindex(index=test_clls, columns=drugs_list)  # Reindex to ensure consistency
            test_pivot = test_pivot.fillna('')  # Fill missing values with blanks

            train_df = response_df[response_df['ccl_name'].isin(train_clls)]
            train_pivot = train_df.pivot_table(index='ccl_name', columns='cpd_name', values='area_under_curve', aggfunc='mean')
            train_pivot = train_pivot.reindex(index=train_clls, columns=drugs_list)  # Reindex to ensure consistency
            train_pivot = train_pivot.fillna('')  # Fill missing values with blanks
            
            test_pivot.rename_axis('').to_csv('data/ctrp_processed_data/cv_5/test_cv_5_fold_'+str(i)+'.csv', index_label=None)
            test_pivot.rename_axis('').to_csv('data/ctrp_processed_data/cv_5/test_cv_5_fold_'+str(i)+'_max_conc.csv', index_label=None)

            train_pivot.rename_axis('').to_csv('data/ctrp_processed_data/cv_5/train_cv_5_fold_'+str(i)+'.csv', index_label=None)
            train_pivot.rename_axis('').to_csv('data/ctrp_processed_data/cv_5/train_cv_5_fold_'+str(i)+'_max_conc.csv', index_label=None)
            
    
    five_folding_data()
    drugs_smiles = list(drug()['canonical_smiles'])
    drugs_names = drug()['drug_name']
    padded_tokenized, max_len_tokens = tokenize_smiles(smiles_list=drugs_smiles)
    drug_smiles_data_df = pd.DataFrame(padded_tokenized, 
                       columns=['token_'+str(i+1) for i in range(max_len_tokens)], 
                       index = drugs_names).rename_axis('drug')
    drug_smiles_data_df.to_csv('data/ctrp_processed_data/dicts_from_csvs/drug_smiles_data.csv')


ctrp_to_gdsc_format()