# Within-dataset experiments

CUDA_VISIBLE_DEVICES=1 python -u DM.py -r 5 -cd oha_VEGETABLE_FREQUENCY_all_otu.csv -cl oha_VEGETABLE_FREQUENCY_binary34_labels.csv -pd oha_AGP_extended_veg_all_otu.csv --ae -dm 128,256,512 -m rf -dt float16 -pt 20 --numJobs 3 -lr 0.001 -htid host_to_indices_VEGETABLE_FREQUENCY_512 &
CUDA_VISIBLE_DEVICES=0 python -u DM.py -r 5 -cd oha_VEGETABLE_FREQUENCY_all_otu.csv -cl oha_VEGETABLE_FREQUENCY_binary34_labels.csv -pd oha_AGP_extended_veg_all_otu.csv --ae -dm 512,1024 -m rf -dt float16 -pt 20 --numJobs 3 -lr 0.001 -htid host_to_indices_VEGETABLE_FREQUENCY_512 &


CUDA_VISIBLE_DEVICES=2 python -u DM.py -r 5 -cd oha_FRUIT_FREQUENCY_all_otu.csv -cl oha_FRUIT_FREQUENCY_binary34_labels.csv -pd oha_AGP_extended_fruit_all_otu.csv --ae -dm 512,1024 -m rf -dt float16 -pt 20 --numJobs 3 -lr 0.001 -htid host_to_indices_FRUIT_FREQUENCY_512 & 
wait
CUDA_VISIBLE_DEVICES=1 python -u DM.py -r 5 -cd oha_FRUIT_FREQUENCY_all_otu.csv -cl oha_FRUIT_FREQUENCY_binary34_labels.csv -pd oha_AGP_extended_fruit_all_otu.csv --ae -dm 128,256,512 -m rf -dt float16 -pt 20 --numJobs 3 -lr 0.001 -htid host_to_indices_FRUIT_FREQUENCY_512 & 


CUDA_VISIBLE_DEVICES=0 python -u DM.py -r 5 -cd oha_total_IBD_otu.csv -cl oha_total_IBD_label.csv -pd oha_AGP_extended_ibd_all_otu.csv --ae -dm 512,1024 -m rf -dt float16 -pt 20 --numJobs 3 -lr 0.001 -htid host_to_indices_total_IBD_512 & 
CUDA_VISIBLE_DEVICES=2 python -u DM.py -r 5 -cd oha_total_IBD_otu.csv -cl oha_total_IBD_label.csv -pd oha_AGP_extended_ibd_all_otu.csv --ae -dm 128,256,512 -m rf -dt float16 -pt 20 --numJobs 3 -lr 0.001 -htid host_to_indices_total_IBD_512 & 
wait


# Cross-generalization experiments
CUDA_VISIBLE_DEVICES=1 python DM.py -r 3 -cd oha_total_IBD_otu.csv -cl oha_total_IBD_label.csv -pd oha_AGP_extended_ibd_all_otu.csv --ae -dm 128,256,512 -m rf -dt float16 -pt 20 --numJobs 3 -lr 0.001 -ad oha_halfvarson_otu.csv,oha_schirmer_IBD_otu.csv -al oha_halfvarson_IBD_labels.csv,oha_schirmer_IBD_labels.csv --use_all_data_for_train &

CUDA_VISIBLE_DEVICES=2 python DM.py -r 5 -cd oha_total_IBD_otu.csv -cl oha_total_IBD_label.csv -pd oha_AGP_extended_ibd_all_otu.csv --ae -dm 512,1024 -m rf -dt float16 -pt 20 --numJobs 3 -lr 0.001 -ad oha_halfvarson_otu.csv,oha_schirmer_IBD_otu.csv -al oha_halfvarson_IBD_labels.csv,oha_schirmer_IBD_labels.csv --use_all_data_for_train