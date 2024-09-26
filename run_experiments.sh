#!/bin/bash
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring paccmann -loss approx_ndcg -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 0 -flag_redo True -tf_record_dir data/ctrp_processed/tfrecords -drug_type_path data/ctrp_processed_data/drugs.txt -pred_dir results/preds_ctrp -model_dir ctrp_ranking_model_dir/ -data_dir data/ctrp_processed_data/ -cv_split_dir data/ctrp_processed_data/

python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring paccmann -loss approx_ndcg -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 0 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring paccmann -loss approx_ndcg -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 1 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring paccmann -loss approx_ndcg -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 2 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring paccmann -loss approx_ndcg -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 3 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring paccmann -loss approx_ndcg -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 4 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring paccmann -loss mse -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 0 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring paccmann -loss mse -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 1 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring paccmann -loss mse -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 2 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring paccmann -loss mse -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 3 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring paccmann -loss mse -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 4 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring nn_baseline -loss approx_ndcg -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 0 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring nn_baseline -loss approx_ndcg -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 1 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring nn_baseline -loss approx_ndcg -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 2 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring nn_baseline -loss approx_ndcg -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 3 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring nn_baseline -loss approx_ndcg -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 4 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring nn_baseline -loss mse -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 0 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring nn_baseline -loss mse -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 1 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring nn_baseline -loss mse -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 2 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring nn_baseline -loss mse -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 3 -flag_redo True
python run_gdsc_model_cv.py -gpu 0 -infix max_conc -scoring nn_baseline -loss mse -gene_feature paccmann -model_suffix 20210617 -cell_wise True -fold_nr 4 -flag_redo True
