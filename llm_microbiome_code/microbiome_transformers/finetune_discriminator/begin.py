import argparse
from torch.utils.data import DataLoader
import numpy as np
import pickle
import torch
from pretrain_hf import ELECTRATrainer
from dataset import ELECTRADataset,create_class_weights,create_weighted_sampler
from transformers import ElectraConfig,ElectraForSequenceClassification
from electra_discriminator import ElectraDiscriminator
from sklearn.model_selection import KFold


def train():
    """
    Essentially the main method for this script.
    Constructs and trains an ELECTRA model for microbiome sample classification.

    This function sets up the datasets, dataloaders, model configuration, and trainer.
    It then runs the training process and returns the results.

    Args:
        train_samples (np.ndarray): Training samples.
        test_samples (np.ndarray): Test samples.
        train_labels (np.ndarray): Labels for training samples.
        test_labels (np.ndarray): Labels for test samples.
        log_file (str): Path to the log file for saving training progress.
        val_samples (np.ndarray, optional): Validation samples. Defaults to None.
        val_labels (np.ndarray, optional): Labels for validation samples. Defaults to None.
        return_embeddings (bool, optional): If True, return embeddings instead of training. Defaults to False.
        use_static_embeddings (bool, optional): If True, use static embeddings. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - train_scores (list): List of training scores for each epoch.
            - train_labels (list): List of training labels for each epoch.
            - train_indices (list): List of training indices for each epoch.
            - test_scores (list): List of test scores for each epoch.
            - test_labels (list): List of test labels.
            - val_scores (list): List of validation scores for each epoch (if validation data provided).
            - val_labels (list): List of validation labels (if validation data provided).

    Note:
        This function uses global variables from the argparse namespace 'args' for various configurations.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--samples", required=True, type=str, help="microbiome samples")
    parser.add_argument("-tl", "--sample_labels",required=False,type=str,default=None, help="labels for samples")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with electra-vocab")
    parser.add_argument("-o", "--output_path", required=False, type=str, help="path to save models ex)output/")
    parser.add_argument("--test_samples",required=False,type=str,help="microbiome samples for test set, only provide if not trying to do cross validation. If providing this, --samples should provide the path to the training samples")
    parser.add_argument("--test_labels",required=False,type=str,help="labels for test set, only provide if not wanting to perform cross validation. If provided --sample_labels should provide the path to the train labels")
    parser.add_argument("--val_samples",required=False,type=str,help="microbiome samples for validation set, only provide if not trying to do cross validation. If providing this, --samples should provide the path to the training samples")
    parser.add_argument("--val_labels",required=False,type=str,help="labels for validation set, only provide if not wanting to perform cross validation. If provided --sample_labels should provide the path to the train labels")    
    parser.add_argument("--path_to_hosts_mapping",required=False,type=str,help="path to pickle file containing a dictionary that maps from host ids to a list of sample ids that belong to that host. Used to split data into train/test/val sets that don't overlap in hosts.")

    parser.add_argument("-hs", "--hidden", type=int, default=100, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=5, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=10, help="number of attention heads")
    parser.add_argument("-s", "--seq_len", type=int, default=1898, help="maximum sequence len")

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--freeze_opt", type=int, default=0, help="parameter for choosing whether to freeze embeds or not, 0 means no freeze, 1 means all embeds are frozen, 2 means all embeds except cls are frozen")
    parser.add_argument("--freeze_encoders", type=int, default=0, help="parameter for choosing how many encoder layers to freeze")    


    parser.add_argument("--cuda", dest='with_cuda', action='store_true',help="train with CUDA")
    parser.add_argument("--no_cuda",dest='with_cuda',action='store_false',help="train on CPU")
    parser.set_defaults(with_cuda=False)

    parser.add_argument("--multi", dest='multi', action='store_true',help="training on multiclass problem")
    parser.set_defaults(multi=False)


    parser.add_argument("--ce",dest='loss_func',action='store_const',const='ce',help="train with cross entropy loss")
    parser.add_argument("--mse",dest='loss_func',action='store_const',const='mse',help="train with mean square error loss loss")
    parser.set_defaults(loss_func='mse')
        
    parser.add_argument("--adam",dest='optim',action='store_const',const='adam',help="train with adam optimizer")
    parser.add_argument("--sgd",dest='optim',action='store_const',const='sgd',help="train with sgd")
    parser.set_defaults(optim='sgd')

    parser.add_argument("--weighted_sampler", dest='class_imb_strat', action='store_true',help="use weighted sampler")
    parser.add_argument("--class_weights",dest='class_imb_strat',action='store_false',help="use class weights")
    parser.set_defaults(class_imb_strat=True)    

    parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
    parser.add_argument("--log_file", type=str,default=None,help="log file for performance metrics" )

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    parser.add_argument("--load_disc", type=str, default=None, help="path to saved state_dict of ELECTRA discriminator")
    parser.add_argument("--load_embed", type=str, default=None, help="path to saved state_dict of ELECTRA discriminator embedding layer")
    parser.add_argument("--num_labels", type = int, default = 2, help="number of labels for classification task")
    
    parser.add_argument("--resume_epoch", type=int, default=0, help="epoch to resume training at")
    parser.add_argument("--repeat", type=int, default=1, help="number of times to repeat training with different data splits")
    parser.add_argument("--ensemble_repeat", type=int, default=0, help="number of times to repeat training with the same data splits for ensemble training")
    parser.add_argument("--data_split_seed", type=int, default=42, help="seed for data split")
    parser.add_argument("--val_split_ensemble_data", action='store_true', help="if true, then we use an 80/10/10 split for train/val/test during the ensemble training")
    parser.add_argument("--return_embeddings", action='store_true', help="if true, then we get the embeddings of the samples and don't train a model")
    parser.add_argument("--n_splits", type=int, default=5, help="number of splits to use for stratified cross-validation")
    parser.add_argument("--cross_gen_test", action='store_true', help="if true, then we perform a cross-dataset generalization test")
    parser.add_argument("--val_split_cross_gen_frac", type=float, default=0.0, help="fraction of the generalization test data to use for validation")
    parser.add_argument("--use_static_embeddings", action='store_true', help="if true, then we use static embeddings and don't train the embeddings")
    parser.add_argument("--val_then_no_val_run", action='store_true', help="if true, then we perform a training run with a validation set and test set, and then a second training run where we merge the val data into the train data, but keep the test set the same")
    parser.add_argument("--val_then_no_val_splits_frac", type=float, nargs=3, default=[0.7, 0.1, 0.2], help="fractions of the data to use for the train, val and test sets during the val then no val run")
    args = parser.parse_args()
    

    # Print the arguments
    print(args)

    samples = np.load(args.samples)
    if not args.return_embeddings:
        labels = np.load(args.sample_labels)
    else:
        labels = np.zeros(samples.shape[0])


    def train_constructor(train_samples,test_samples,train_labels,test_labels,log_file,val_samples=None,val_labels=None,return_embeddings=False,use_static_embeddings=False):
        """
        Constructs and trains an ELECTRA model for microbiome sample classification.

        This function sets up the datasets, dataloaders, model configuration, and trainer.
        It then runs the training process and returns the results.

        Args:
            train_samples (np.ndarray): Training samples.
            test_samples (np.ndarray): Test samples.
            train_labels (np.ndarray): Labels for training samples.
            test_labels (np.ndarray): Labels for test samples.
            log_file (str): Path to the log file for saving training progress.
            val_samples (np.ndarray, optional): Validation samples. Defaults to None.
            val_labels (np.ndarray, optional): Labels for validation samples. Defaults to None.
            return_embeddings (bool, optional): If True, return embeddings instead of training. Defaults to False.
            use_static_embeddings (bool, optional): If True, use static embeddings. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - train_scores (list): List of training scores for each epoch.
                - train_labels (list): List of training labels for each epoch.
                - train_indices (list): List of training indices for each epoch.
                - test_scores (list): List of test scores for each epoch.
                - test_labels (list): List of test labels.
                - val_scores (list): List of validation scores for each epoch (if validation data provided).
                - val_labels (list): List of validation labels (if validation data provided).

        Note:
            This function uses global variables from the argparse namespace 'args' for various configurations.
        """
        print("Loading Train Dataset")
        train_dataset = ELECTRADataset(train_samples, args.vocab_path,train_labels)

        print("Loading Test Dataset")
        test_dataset = ELECTRADataset(test_samples, args.vocab_path,test_labels)

        val_data_loader = None
        if val_samples is not None and val_labels is not None:
            val_dataset = ELECTRADataset(val_samples,args.vocab_path,val_labels)
            val_data_loader = DataLoader(val_dataset, batch_size=1, num_workers=args.num_workers)                
        
        class_weights = None

        print("Creating Dataloader")


        if args.class_imb_strat:
            sampler = create_weighted_sampler(train_labels)
            train_data_loader = DataLoader(train_dataset,sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers)
        else:
            class_weights = create_class_weights(train_labels)
            train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=True)

        train_orig_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,shuffle=False)
        test_data_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers)


        vocab_len = train_dataset.vocab_len()
    

        electra_config = ElectraConfig(vocab_size=vocab_len,embedding_size=args.hidden,hidden_size=args.hidden*2,num_hidden_layers=args.layers,num_attention_heads=args.attn_heads,intermediate_size=4*args.hidden,max_position_embeddings=args.seq_len,num_labels=args.num_labels)
        electra = ElectraDiscriminator(electra_config,torch.from_numpy(train_dataset.embeddings),args.load_disc,args.load_embed,use_static_embeddings=use_static_embeddings)
        print(electra)
        #pdb.set_trace()
        print("Creating Electra Trainer")
        if args.class_imb_strat:
            trainer = ELECTRATrainer(electra, vocab_len, train_dataloader=train_data_loader,train_orig_dataloader = train_orig_dataloader, test_dataloader=test_data_loader,val_dataloader=val_data_loader,
                                lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                                with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,log_file=log_file,
                                freeze_embed=args.freeze_opt,freeze_encoders=args.freeze_encoders,loss_func=args.loss_func,optim=args.optim,hidden_size=args.hidden*2,use_static_embeddings=use_static_embeddings)
        else:
            trainer = ELECTRATrainer(electra, vocab_len, train_dataloader=train_data_loader,train_orig_dataloader = train_orig_dataloader, test_dataloader=test_data_loader,val_dataloader=val_data_loader,
                            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq,log_file=log_file,
                            freeze_embed=args.freeze_opt,freeze_encoders=args.freeze_encoders,class_weights=torch.tensor(class_weights,dtype=torch.float),loss_func=args.loss_func,optim=args.optim,hidden_size=args.hidden*2,use_static_embeddings=use_static_embeddings)

        if return_embeddings:
            print("Getting embeddings")
            return trainer.get_embeddings(train_orig_dataloader,multi=args.multi)
        print("Training Start")
        for epoch in range(args.resume_epoch,args.epochs+args.resume_epoch):
            #pdb.set_trace()
            trainer.train(epoch,args.multi)
            #if epoch == 4 or epoch == 9 or epoch == 14 or epoch == 19:
                #trainer.save(epoch, args.output_path)
                #pdb.set_trace()

            trainer.train_orig_dist(epoch,args.multi)
            if val_samples is not None and val_labels is not None:
                trainer.val(epoch,args.multi)
            trainer.test(epoch,args.multi)
        train_scores = trainer.train_scores
        test_scores = trainer.test_scores
        val_scores = trainer.val_scores
        train_labels = trainer.train_labels
        test_labels = trainer.test_labels
        val_labels = trainer.val_labels
        train_indices = trainer.train_indices
        return train_scores, train_labels, train_indices, test_scores, test_labels, val_scores, val_labels
    

    if args.sample_labels is None and args.return_embeddings:
        # Just embed the data
        embeddings = train_constructor(samples,samples,labels,labels,args.log_file,return_embeddings=True)
        torch.save(torch.tensor(embeddings), args.output_path)
    elif args.path_to_hosts_mapping is not None and args.ensemble_repeat == 0 and args.val_then_no_val_run:
        print("Running val then no val run")
        host_mapping = pickle.load(open(args.path_to_hosts_mapping, 'rb'))
        # First, we split the data into train / val / test sets with no overlap in hosts and the appropriate class balance. We aim for ratio of data determined by args.val_then_no_val_splits_frac, with a +/- 10% tolerance for both the class balance and the number of samples.
        # Then, we run the training with the val set, and then merge the val set into the train set and run the training again.        
        all_hosts = list(host_mapping.keys())
        target_class_balance = np.mean(labels)
        target_train_frac, target_val_frac, target_test_frac = args.val_then_no_val_splits_frac

        def get_host_split(hosts, seed):
            """
            Split the list of hosts into train, validation, and test sets.

            Args:
                hosts (list): List of all host IDs.
                seed (int): Random seed for reproducibility.

            Returns:
                tuple: Three lists of host IDs for train, validation, and test sets.
            """
            rng = np.random.default_rng(seed)
            rng.shuffle(hosts)
            n_hosts = len(hosts)
            train_split = int(n_hosts * target_train_frac)
            val_split = int(n_hosts * (target_train_frac + target_val_frac))
            return hosts[:train_split], hosts[train_split:val_split], hosts[val_split:]

        def get_samples_and_labels(host_list):
            """
            Retrieve samples and labels for a given list of hosts.

            Args:
                host_list (list): List of host IDs.

            Returns:
                tuple: Two numpy arrays containing samples and their corresponding labels.
            """
            indices = [i for host in host_list for i in host_mapping[host]]
            return samples[indices], labels[indices]

        max_attempts = 10000
        for attempt in range(max_attempts):
            seed = hash((args.data_split_seed, attempt)) % 2**32 - 1
            train_hosts, val_hosts, test_hosts = get_host_split(all_hosts, seed)
            
            train_samples, train_labels = get_samples_and_labels(train_hosts)
            val_samples, val_labels = get_samples_and_labels(val_hosts)
            test_samples, test_labels = get_samples_and_labels(test_hosts)
            
            train_balance = np.mean(train_labels)
            val_balance = np.mean(val_labels)
            test_balance = np.mean(test_labels)
            
            train_frac = len(train_samples) / len(samples)
            val_frac = len(val_samples) / len(samples)
            test_frac = len(test_samples) / len(samples)
            
            if (abs(train_balance - target_class_balance) <= 0.1 * target_class_balance and
                abs(val_balance - target_class_balance) <= 0.1 * target_class_balance and
                abs(test_balance - target_class_balance) <= 0.1 * target_class_balance and
                abs(train_frac - target_train_frac) <= 0.1 * target_train_frac and
                abs(val_frac - target_val_frac) <= 0.1 * target_val_frac and
                abs(test_frac - target_test_frac) <= 0.1 * target_test_frac):
                break
        else:
            raise ValueError("Could not find a satisfactory split after maximum attempts")

        print(f"Split achieved after {attempt + 1} attempts")
        print(f"Train samples: {len(train_samples)}, balance: {train_balance:.4f}")
        print(f"Val samples: {len(val_samples)}, balance: {val_balance:.4f}")
        print(f"Test samples: {len(test_samples)}, balance: {test_balance:.4f}")

        # Run with validation set
        log_file_with_val = f"{args.log_file}_with_val.txt"
        returned_train_scores_with_val, returned_train_labels_with_val, returned_train_indices_with_val, returned_test_scores_with_val, returned_test_labels_with_val, returned_val_scores_with_val, returned_val_labels_with_val = train_constructor(
            train_samples, test_samples, train_labels, test_labels, log_file_with_val, val_samples, val_labels, use_static_embeddings=args.use_static_embeddings
        )

        # Run without validation set (merge train and val)
        train_val_samples = np.concatenate((train_samples, val_samples))
        print("train_samples.shape", train_samples.shape)
        print("val_samples.shape", val_samples.shape)
        print("train_val_samples.shape", train_val_samples.shape)
        print("train_labels.shape", train_labels.shape)
        print("val_labels.shape", val_labels.shape)
        train_val_labels = np.concatenate((train_labels, val_labels))
        log_file_without_val = f"{args.log_file}_without_val.txt"
        returned_train_scores_without_val, returned_train_labels_without_val, returned_train_indices_without_val, returned_test_scores_without_val, returned_test_labels_without_val, _, _ = train_constructor(
            train_val_samples, test_samples, train_val_labels, test_labels, log_file_without_val, use_static_embeddings=args.use_static_embeddings
        )

        # Compute and save performance metrics
        def compute_metrics(scores, labels):
            auc = ELECTRATrainer.calc_auc(labels, scores)
            aupr = ELECTRATrainer.calc_aupr(labels, scores)
            neg_aupr = ELECTRATrainer.calc_aupr(1 - labels, scores)
            return auc, aupr, neg_aupr

        with open(f"{args.log_file}_val_then_no_val_performance.txt", "w") as f:
            f.write("Run,Epoch,Train AUC,Val AUC,Test AUC,Train AUPR,Val AUPR,Test AUPR,Train Neg AUPR,Val Neg AUPR,Test Neg AUPR\n")
            
            for epoch in range(len(returned_train_scores_with_val)):
                train_metrics = compute_metrics(returned_train_scores_with_val[epoch], returned_train_labels_with_val[epoch])
                val_metrics = compute_metrics(returned_val_scores_with_val[epoch], val_labels)
                test_metrics = compute_metrics(returned_test_scores_with_val[epoch], test_labels)
                
                f.write(f"With Val,{epoch},{train_metrics[0]:.4f},{val_metrics[0]:.4f},{test_metrics[0]:.4f},")
                f.write(f"{train_metrics[1]:.4f},{val_metrics[1]:.4f},{test_metrics[1]:.4f},")
                f.write(f"{train_metrics[2]:.4f},{val_metrics[2]:.4f},{test_metrics[2]:.4f}\n")

            for epoch in range(len(returned_train_scores_without_val)):
                train_metrics = compute_metrics(returned_train_scores_without_val[epoch], returned_train_labels_without_val[epoch])
                test_metrics = compute_metrics(returned_test_scores_without_val[epoch], test_labels)
                
                f.write(f"Without Val,{epoch},{train_metrics[0]:.4f},N/A,{test_metrics[0]:.4f},")
                f.write(f"{train_metrics[1]:.4f},N/A,{test_metrics[1]:.4f},")
                f.write(f"{train_metrics[2]:.4f},N/A,{test_metrics[2]:.4f}\n")

        print(f"Performance metrics saved to {args.log_file}_val_then_no_val_performance.txt")
        

    elif args.test_samples is not None and args.test_labels is not None and args.val_samples is None and args.val_labels is None and args.repeat == 1 and not args.cross_gen_test and not args.val_split_cross_gen_frac > 0:
        print("Simple train / test split")
        test_samples = np.load(args.test_samples)
        test_labels = np.load(args.test_labels)
        #pdb.set_trace()
        log_file = args.log_file+".txt"
        train_constructor(samples,test_samples,labels,test_labels,log_file)
    
    elif args.test_samples is not None and args.test_labels is not None and args.val_samples is None and args.val_labels is None and args.ensemble_repeat > 0 and args.cross_gen_test and not args.val_split_cross_gen_frac > 0:
        print("Running ensembled cross-gen test")
        test_samples = np.load(args.test_samples)
        test_labels = np.load(args.test_labels)

        ensemble_scores_train = []
        ensemble_indices_train = []
        ensemble_labels_train = []

        ensemble_scores_test = []
        ensemble_labels_test = []
        for ensemble_run in range(args.ensemble_repeat):
            print(f"Train samples: {len(samples)}")
            print(f"Test samples: {len(test_samples)}")

            print(f"Running ensemble iteration {ensemble_run + 1}/{args.ensemble_repeat}")
            log_file = f"{args.log_file}_ensemble{ensemble_run + 1}.txt"
            
            # Train the model and get predictions
            returned_train_scores, returned_train_labels, returned_train_indices, returned_test_scores, returned_test_labels, _, _ = train_constructor(samples, test_samples, labels, test_labels, log_file, use_static_embeddings=args.use_static_embeddings)

            print("returned_train_scores.shape", np.array(returned_train_scores).shape)
            print("returned_test_scores.shape", np.array(returned_test_scores).shape)
            ensemble_scores_train.append(returned_train_scores)
            ensemble_scores_test.append(returned_test_scores)
            ensemble_labels_train.append(returned_train_labels)
            ensemble_labels_test.append(returned_test_labels)
            ensemble_indices_train.append(returned_train_indices)
        # After all runs, combine predictions (using simple averaging) and score the results
        # Ensemble scores are of shape (ensemble_repeat, num_epochs, num_samples)
        ensemble_scores_train = np.array(ensemble_scores_train)
        ensemble_scores_test = np.array(ensemble_scores_test)

        # Average scores are of shape (num_epochs, num_samples)
        average_scores_test = np.mean(ensemble_scores_test, axis=0)

        average_scores_train = np.zeros((ensemble_scores_train[0].shape))
        for run_scores, run_indices in zip(ensemble_scores_train, ensemble_indices_train):
            for epoch_num, (epoch_scores, epoch_indices) in enumerate(zip(run_scores, run_indices)):
                for index in epoch_indices:
                    average_scores_train[epoch_num][index] += epoch_scores[index]
        average_scores_train = average_scores_train / args.ensemble_repeat
        

        print("average_scores_train.shape", average_scores_train.shape)
        print("average_scores_test.shape", average_scores_test.shape)
        
        all_runs_aucs_train = []
        all_runs_aucs_test = []
        all_runs_auprs_train = []
        all_runs_auprs_test = []
        all_runs_neg_auprs_train = []
        all_runs_neg_auprs_test = []
        all_runs_run_nums = []
        all_runs_epoch_nums = []
        # For each run, print the AUC and AUPR for each epoch
        print("Run | Epoch | Train AUC | Test AUC | Train AUPR | Test AUPR | Train Neg AUPR | Test Neg AUPR")
        
        print("-" * 80)
        for run in range(len(ensemble_scores_train)):
            for epoch in range(len(ensemble_scores_train[run])):
                train_auc = ELECTRATrainer.calc_auc(ensemble_labels_train[run][epoch], ensemble_scores_train[run][epoch])
                test_auc = ELECTRATrainer.calc_auc(ensemble_labels_test[run][epoch], ensemble_scores_test[run][epoch])
                train_aupr = ELECTRATrainer.calc_aupr(ensemble_labels_train[run][epoch], ensemble_scores_train[run][epoch])
                test_aupr = ELECTRATrainer.calc_aupr(ensemble_labels_test[run][epoch], ensemble_scores_test[run][epoch])
                train_neg_aupr = ELECTRATrainer.calc_aupr(1-ensemble_labels_train[run][epoch], ensemble_scores_train[run][epoch])
                test_neg_aupr = ELECTRATrainer.calc_aupr(1-ensemble_labels_test[run][epoch], ensemble_scores_test[run][epoch])
                print(f"{run:5d} | {epoch:5d} | {train_auc:.4f} | {test_auc:.4f} | {train_aupr:.4f} | {test_aupr:.4f} | {train_neg_aupr:.4f} | {test_neg_aupr:.4f}")
                all_runs_aucs_train.append(train_auc)
                all_runs_aucs_test.append(test_auc)
                all_runs_auprs_train.append(train_aupr)
                all_runs_auprs_test.append(test_aupr)
                all_runs_neg_auprs_train.append(train_neg_aupr)
                all_runs_neg_auprs_test.append(test_neg_aupr)
                all_runs_run_nums.append(run)
                all_runs_epoch_nums.append(epoch)
        
        # Save the performance metrics in one file
        with open(f"{args.log_file}_ensemble_per_epoch_performance.txt", "w") as f:
            f.write("Run,Epoch,Train AUC,Test AUC,Train AUPR,Test AUPR,Train Neg AUPR,Test Neg AUPR\n")
            for i, (run, epoch) in enumerate(zip(all_runs_run_nums, all_runs_epoch_nums)):
                f.write(f"{run},{epoch},{all_runs_aucs_train[i]:.4f},{all_runs_aucs_test[i]:.4f},{all_runs_auprs_train[i]:.4f},{all_runs_auprs_test[i]:.4f},{all_runs_neg_auprs_train[i]:.4f},{all_runs_neg_auprs_test[i]:.4f}\n")


        # Score the average scores using AUC and AUPR for all epochs
        ensemble_epoch_aucs_train = []
        ensemble_epoch_aucs_test = []
        ensemble_epoch_auprs_train = []
        ensemble_epoch_auprs_test = []
        ensemble_epoch_neg_auprs_train = []
        ensemble_epoch_neg_auprs_test = []
        for epoch in range(len(average_scores_train)):
            auc_train = ELECTRATrainer.calc_auc(labels, average_scores_train[epoch])
            auc_test = ELECTRATrainer.calc_auc(test_labels, average_scores_test[epoch])
            aupr_train = ELECTRATrainer.calc_aupr(labels, average_scores_train[epoch])
            aupr_test = ELECTRATrainer.calc_aupr(test_labels, average_scores_test[epoch])
            neg_aupr_train = ELECTRATrainer.calc_aupr(1-labels, average_scores_train[epoch])
            neg_aupr_test = ELECTRATrainer.calc_aupr(1-test_labels, average_scores_test[epoch])
            ensemble_epoch_aucs_train.append(auc_train)
            ensemble_epoch_aucs_test.append(auc_test)
            ensemble_epoch_auprs_train.append(aupr_train)
            ensemble_epoch_auprs_test.append(aupr_test)
            ensemble_epoch_neg_auprs_train.append(neg_aupr_train)
            ensemble_epoch_neg_auprs_test.append(neg_aupr_test)
        
        print("Epoch | Train AUC | Test AUC | Train AUPR | Test AUPR | Train Neg AUPR | Test Neg AUPR")
        print("-" * 80)
        for epoch in range(len(ensemble_epoch_aucs_train)):
            print(f"{epoch:5d} | {ensemble_epoch_aucs_train[epoch]:.4f} | {ensemble_epoch_aucs_test[epoch]:.4f} | {ensemble_epoch_auprs_train[epoch]:.4f} | {ensemble_epoch_auprs_test[epoch]:.4f} | {ensemble_epoch_neg_auprs_train[epoch]:.4f} | {ensemble_epoch_neg_auprs_test[epoch]:.4f}")

        # Save the performance metrics in one file
        with open(f"{args.log_file}_ensemble_performance.txt", "w") as f:
            f.write("Epoch,Train AUC,Test AUC,Train AUPR,Test AUPR,Train Neg AUPR,Test Neg AUPR\n")
            for epoch in range(len(ensemble_epoch_aucs_train)):
                f.write(f"{epoch},{ensemble_epoch_aucs_train[epoch]:.4f},{ensemble_epoch_aucs_test[epoch]:.4f},{ensemble_epoch_auprs_train[epoch]:.4f},{ensemble_epoch_auprs_test[epoch]:.4f},{ensemble_epoch_neg_auprs_train[epoch]:.4f},{ensemble_epoch_neg_auprs_test[epoch]:.4f}\n")
    
    elif args.test_samples is not None and args.test_labels is not None and args.val_samples is None and args.val_labels is None and args.ensemble_repeat == 0 and args.cross_gen_test and args.val_split_cross_gen_frac > 0:
        print("Running NON-ensembled cross-gen test with validation split")

        test_samples = np.load(args.test_samples)
        test_labels = np.load(args.test_labels)

        # Split the train samples into train and val, leaving test alone
        # Set the random seed for reproducibility
        np.random.seed(args.data_split_seed)

        # Calculate the number of samples for validation
        val_size = int(len(samples) * args.val_split_cross_gen_frac)

        # Randomly shuffle the indices
        indices = np.arange(len(samples))
        np.random.shuffle(indices)

        # Split the indices
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        # Create the validation set
        val_samples = samples[val_indices]
        val_labels = labels[val_indices]

        # Update the training set
        samples = samples[train_indices]
        labels = labels[train_indices]

        print(f"Train set size: {len(samples)}")
        print(f"Validation set size: {len(val_samples)}")
        print(f"Test set size: {len(test_samples)}")

        # Run the training and record the results
        log_file = args.log_file + "_cross_gen_test.txt"
        returned_train_scores, returned_train_labels, returned_train_indices, returned_test_scores, returned_test_labels, returned_val_scores, returned_val_labels = train_constructor(samples, test_samples, labels, test_labels, log_file, val_samples, val_labels)

        # Compute and save train/val/test AUC, AUPR, and Neg AUPR for all epochs
        epoch_metrics = []
        for epoch in range(len(returned_train_scores)):
            train_auc = ELECTRATrainer.calc_auc(returned_train_labels[epoch], returned_train_scores[epoch])
            val_auc = ELECTRATrainer.calc_auc(returned_val_labels[epoch], returned_val_scores[epoch])
            test_auc = ELECTRATrainer.calc_auc(test_labels, returned_test_scores[epoch])
            
            train_aupr = ELECTRATrainer.calc_aupr(returned_train_labels[epoch], returned_train_scores[epoch])
            val_aupr = ELECTRATrainer.calc_aupr(returned_val_labels[epoch], returned_val_scores[epoch])
            test_aupr = ELECTRATrainer.calc_aupr(test_labels, returned_test_scores[epoch])
            
            train_neg_aupr = ELECTRATrainer.calc_aupr(1 - returned_train_labels[epoch], returned_train_scores[epoch])
            val_neg_aupr = ELECTRATrainer.calc_aupr(1 - returned_val_labels[epoch], returned_val_scores[epoch])
            test_neg_aupr = ELECTRATrainer.calc_aupr(1 - test_labels, returned_test_scores[epoch])
            
            epoch_metrics.append([epoch, train_auc, val_auc, test_auc, train_aupr, val_aupr, test_aupr, train_neg_aupr, val_neg_aupr, test_neg_aupr])
        
        # Save the performance metrics
        performance_file = f"{args.log_file}_cross_gen_performance.txt"
        with open(performance_file, "w") as f:
            f.write("Epoch,Train AUC,Val AUC,Test AUC,Train AUPR,Val AUPR,Test AUPR,Train Neg AUPR,Val Neg AUPR,Test Neg AUPR\n")
            for metrics in epoch_metrics:
                f.write(",".join(map(str, metrics)) + "\n")
        
        print(f"Performance metrics saved to {performance_file}")

    elif args.test_samples is not None and args.test_labels is not None and args.val_samples is not None and args.val_labels is not None:
        print("Running train / test / val split from specified files")
        test_samples = np.load(args.test_samples)
        test_labels = np.load(args.test_labels)
        val_samples = np.load(args.val_samples)
        val_labels = np.load(args.val_labels)
        #pdb.set_trace()
        log_file = args.log_file+".txt"
        train_constructor(samples,test_samples,labels,test_labels,log_file,val_samples,val_labels)
    else:
        print("Running cross validation on training data without blocking")
        split_count = 1
        kf = KFold(n_splits=args.n_splits,shuffle=True,random_state=args.data_split_seed)
        for train_index,test_index in kf.split(samples):
            log_file = args.log_file+"_valset"+str(split_count)+".txt"
            train_samples = samples[train_index]
            train_labels = labels[train_index]
            test_samples = samples[test_index]
            test_labels = labels[test_index]

            train_constructor(train_samples,test_samples,train_labels,test_labels,log_file)
            split_count += 1

if __name__ == "__main__":
    train()