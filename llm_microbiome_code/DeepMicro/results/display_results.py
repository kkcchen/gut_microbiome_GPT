import pandas as pd
import argparse
import matplotlib.pyplot as plt
from terminaltables import AsciiTable
import numpy as np
import os


def main():
    parser = argparse.ArgumentParser(description='Read a CSV file using pandas.')
    parser.add_argument('--file', type=str, default="oha_FRUIT_FREQUENCY_all_otu_pretrained_result.txt", help='Path to the CSV file to be read')
    #parser.add_argument('--file', type=str, default="oha_FRUIT_FREQUENCY_all_otu_result.txt", help='Path to the CSV file to be read')
    #parser.add_argument('--file', type=str, default="oha_total_IBD_otu_pretrained_result.txt", help='Path to the CSV file to be read')
    #parser.add_argument('--file', type=str, default="oha_total_IBD_otu_result.txt", help='Path to the CSV file to be read')
    #parser.add_argument('--file', type=str, default="oha_total_IBD_otu_pretrained_test_inv_result.txt", help='Path to the CSV file to be read')
    #parser.add_argument('--file', type=str, default="oha_VEGETABLE_FREQUENCY_all_otu_pretrained_result.txt", help='Path to the CSV file to be read')
    #parser.add_argument('--file', type=str, default="oha_VEGETABLE_FREQUENCY_all_otu_result.txt", help='Path to the CSV file to be read')

    parser.add_argument('--save_loc', type=str, default=None, help='Path to the location in which we save the plot')
    parser.add_argument('--csv_start_loc', type=int, default=0, help='The row in the csv file where the data starts')
    parser.add_argument('--tt_print', action='store_true', help='Print the table to the terminal')
    parser.add_argument('--lt_print', action='store_true', help='Print the table in LaTeX format')
    args = parser.parse_args()
    if args.save_loc is None:
        args.save_loc = args.file.replace(".txt", "_plots")
    if not os.path.exists(args.save_loc):
        os.makedirs(args.save_loc)

    try:
        df = pd.read_csv(args.file, header=None, usecols=[0, 1, 2, 3, 4, 5, 6, 7], skiprows=args.csv_start_loc)
        df.columns = ["desc", "auc", "acc", "recall", "precision", "f1", "aupr", "neg aupr"]
        has_neg_aupr = True
        float(df['neg aupr'].iloc[0])
    except:
        df = pd.read_csv(args.file, header=None, usecols=[0, 1, 2, 3, 4, 5, 6], skiprows=args.csv_start_loc)
        df.columns = ["desc", "auc", "acc", "recall", "precision", "f1", "aupr"]
        has_neg_aupr = False

    df['desc'] = df['desc'].str.replace("_one_hot_abundances_fracs.csv", "")
    df['desc'] = df['desc'].str.replace("DAE", "AE")
    df['desc'] = df['desc'].str.replace("rfsh", "rf_sh")
    df['desc'] = df['desc'].str.replace("rfhf", "rf_hf")

    df['desc'] = df['desc'].str.replace("svmsh", "svm_sh")
    df['desc'] = df['desc'].str.replace("svmhf", "svm_hf")

    df['desc'] = df['desc'].str.replace("mlpsh", "mlp_sh")
    df['desc'] = df['desc'].str.replace("mlphf", "mlp_hf")

    filter_strs = ["host_split"]
    for filter_str in filter_strs:
        df = df[df['desc'].str.contains(filter_str)]
        df['desc'] = df['desc'].str.replace(filter_str, "")


    print(df)
    duplicate_rows = df[df.duplicated()]
    print(f"Number of duplicate rows: {duplicate_rows.shape[0]}")
    if duplicate_rows.shape[0] > 0:
        print("Duplicate Rows:")
        print(duplicate_rows)
    
    # Metrics to plot
    metrics = ["auc", "acc", "recall", "precision", "f1", "aupr"]
    metrics = ["auc", "aupr"]
    if has_neg_aupr:
        metrics.append("neg aupr")

    architectures = ["CAE", "VAE", "AE", "PCA", "RandP", "rf", "EMB_cls", "EMB_avg", "EMB_weighted", "EMB_embed__weighted", "EMB_embed__avg", "EMB_embed__log_weighted"]
    #architectures = ["CAE", "AE", "PCA", "RandP"]

    if "_fruit" in str.lower(args.file):
        datasets = ["Fruit"]
    elif "_veg" in str.lower(args.file):
        datasets = ["Veg"]
    else:
        datasets = ["IBD", "Schirmer", "Halfvarson"]

    # Iterate over each metric and create a separate plot
    for metric in metrics:
        # Create figure for the current metric
        fig, axs = plt.subplots(len(datasets), len(architectures), figsize=(len(architectures)*10, len(datasets)*10), sharey=True, sharex=False)
        fig.suptitle(f'Comparison of {metric} Across Architectures and Datasets')
        if len(datasets) == 1 or len(architectures) == 1:
            axs = axs.reshape(len(datasets), len(architectures))

        for i, dataset in enumerate(datasets):
            table_data = [["Arch/Hyperparameter"]]
            for dataset in datasets:
                table_data[0].append(dataset)
            for j, architecture in enumerate(architectures):
                if j > 0:  # Add a horizontal line separator between different architectures
                    table_data.append(['-' * (20 if l == 0 else 15) for l in range(len(datasets) + 1)])
                architecture_data = []
                for i, dataset in enumerate(datasets):
                    # Filter rows for the current architecture and dataset
                    if dataset == "IBD":
                        if architecture in ["AE", "rf", "svm", "mlp"]:
                            architecture_df = df[(df['desc'].str.match(f'^{architecture}') | df['desc'].str.contains(f'train{architecture}')) & ~df['desc'].str.contains("_schir") & ~df['desc'].str.contains("_halfvar")]
                        else:
                            architecture_df = df[df['desc'].str.contains(architecture) & ~df['desc'].str.contains("_schir") & ~df['desc'].str.contains("_halfvar")]
                    elif dataset == "Schirmer":
                        if architecture in ["AE", "rf", "svm", "mlp"]:
                            architecture_df = df[(df['desc'].str.match(f'^{architecture}') | df['desc'].str.contains(f'train{architecture}')) & df['desc'].str.contains("_schir")]
                        else:
                            architecture_df = df[df['desc'].str.contains(architecture) & df['desc'].str.contains("_schir")]
                    elif dataset == "Halfvarson":
                        if architecture in ["AE", "rf", "svm", "mlp"]:
                            architecture_df = df[(df['desc'].str.match(f'^{architecture}') | df['desc'].str.contains(f'train{architecture}')) & df['desc'].str.contains("_halfvar")]
                        else:
                            architecture_df = df[df['desc'].str.contains(architecture) & df['desc'].str.contains("_halfvar")]
                    elif dataset in ["Fruit", "Veg"]:
                        if architecture in ["AE", "rf", "svm", "mlp"]:
                            architecture_df = df[df['desc'].str.match(f'^{architecture}') | df['desc'].str.contains(f'train{architecture}')]
                        else:
                            architecture_df = df[df['desc'].str.contains(architecture)]
                    # print("i,j", i, j)
                    # print("architecture_df", architecture_df)
                    # print("architecture", architecture)
                    # print("dataset", dataset)

                    grouped_architecture_df = architecture_df.groupby('desc').agg(['mean', 'var']).fillna(0)
                    num_experiments = architecture_df['desc'].value_counts().to_dict()
                    for desc, count in num_experiments.items():
                        if count < 5:
                            print(f"{desc}: {count} experiments")
                        if count > 5:
                            grouped_architecture_df = grouped_architecture_df.head(5)
                    means = grouped_architecture_df[metric]['mean'].values
                    standard_deviations = np.sqrt(grouped_architecture_df[metric]['var'].values)
                    labels = grouped_architecture_df.index.values

                    # Plotting each architecture and dataset in its subplot
                    axs[i, j].bar(labels, means, yerr=standard_deviations, capsize=5)
                    axs[i, j].set_title(f'{architecture} - {dataset}')
                    axs[i, j].tick_params(labelrotation=45)
                    axs[i, j].set_xlabel('Hyperparameter Setting')
                    axs[i, j].set_ylabel('Value' if j == 0 else '')

                    # Prepare data for the table
                    if i == 0:
                        for label in labels:
                            architecture_data.append([label])
                    for k, (mean, variance) in enumerate(zip(means, standard_deviations)):
                        architecture_data[k].append(f"{mean:.3f}({variance:.2f})".replace("(0.", "(."))

                table_data.extend(architecture_data)

        # Create and display the table for the current metric
        table = AsciiTable(table_data)
        print(f"\n{metric} Comparison Across Architectures and Datasets")
        if args.tt_print:
            print(table.table)

        plt.tight_layout()

        # Save each plot with a unique name based on the metric
        plt.savefig(f"{args.save_loc}_{metric}.png")

        # Convert table data to a LaTeX table format
        latex_table = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|" + "l|"*len(table_data[0]) + "}\n\\hline\n"
        for i, row in enumerate(table_data):
            if i == 0:  # Header row
                latex_table += " & ".join(row) + " \\\\ \\hline\n"
            else:
                latex_table += " & ".join(row) + " \\\\ \\hline\n"
        latex_table += "\\end{tabular}\n\\caption{" + metric + " Comparison Across Architectures and Datasets}\n\\end{table}"
        
        if args.lt_print:
            print(latex_table)

if __name__ == "__main__":
    main()
