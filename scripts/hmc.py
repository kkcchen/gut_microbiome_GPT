import pandas as pd
import numpy as np

class HMC():
    def __init__(
        self,
        taxonomic_table_path: str,
        project_metadata_path: str,
        sample_metadata_path: str,
        tags_path: str,
        load_taxonomic_table_full: bool = False,
        verbose: bool = True,
        ):
        # load taxonomic table
        self._load_taxonomic_table(taxonomic_table_path, load_taxonomic_table_full)
        if verbose:
            print("loaded taxonomic table")
        # load project metadata
        self._load_project_metadata(project_metadata_path)
        if verbose:
            print("loaded project metadata")
        # load sample metadata
        self._load_sample_metadata(sample_metadata_path)
        if verbose:
            print("loaded sample metadata")
        # load tags
        self._load_tags(tags_path)
        if verbose:
            print("loaded tags")
        
    
    def _load_taxonomic_table(self, taxonomic_table_path, load_taxonomic_table_full):
        """
        assumes path leads to csv file
        assigns the attributes 
        - self.taxonomic_table: a numpy array
        - self.sample_list: a pandas series
        - self.n_samples: number of samples
        - self.n_bacterias: number of bacterias (columns)
        """
        if load_taxonomic_table_full:
            df = pd.read_csv(taxonomic_table_path, index_col=0)
            self.sample_list = df["sample"]
            df = df.drop(columns=["sample"])
            self.taxonomic_table = df.to_numpy()
        else:
            df_col = pd.read_csv(taxonomic_table_path, nrows=5, index_col=0)
            num_cols = df_col.drop(columns=["sample"]).shape[1]
            df_row = pd.read_csv(taxonomic_table_path, usecols=[1])
            num_rows = df_row.shape[0]
            self.sample_list = df_row
            self.taxonomic_table = np.zeros((num_rows, num_cols))
        self.n_samples = self.taxonomic_table.shape[0]
        self.n_bacterias = self.taxonomic_table.shape[1]

    def _load_sample_metadata(self, sample_metadata_path):
        """
        assumes path leads to tsv file
        assigns the attribute
        - self.sample_metadata: a pandas dataframe
        """
        df = pd.read_csv(sample_metadata_path, sep='\t')
        df["table_index"] = df["project"] + "_" + df["srr"]
        df = df.set_index("table_index")
        self.sample_metadata = df
        assert self.sample_metadata.shape[0] == self.n_samples
        
    def _load_project_metadata(self, project_metadata_path):
        """
        assumes path leads to csv file
        assigns the attribute
        - self.project_metadata: a pandas dataframe"""
        df = pd.read_csv(project_metadata_path, index_col=0)
        self.project_metadata = df

    def _load_tags(self, tags_path):
        """
        assumes path leads to tsv file
        assigns the attribute
        - self.tags: a pandas dataframe
        - self.tags_index_dict: a dictionary with keys being the table_index and values being a list of indices
                                the indices indicate the rows in the tags dataframe that correspond to the table_index
        """
        df = pd.read_csv(tags_path, sep='\t')
        df["table_index"] = df["project"] + "_" + df["srr"]
        df = df.set_index("table_index")
        # df = df.drop(columns=["project", "srr", "srs"])
        index_dict = df.groupby("table_index").apply(lambda x: list(x.index)).to_dict()
        self.tags = df
        self.tags_index_dict = index_dict

    def summary_statistics(self, sample_count_distribution:bool=True, bacteria_count_distribution:bool=True):
        """
        returns a dictionary with the following keys:
        - sample_count_distribution: a dictionary with keys being the number of bacterias and values being the number of samples
        - bacteria_count_distribution: a dictionary with keys being the number of samples and values being the number of bacterias
        """
        sample_count_distribution_dict = {}
        bacteria_count_distribution_dict = {}
        if sample_count_distribution:
            for i in range(self.n_samples):
                n_bacterias = np.sum(self.taxonomic_table[i] > 0)
                if n_bacterias not in sample_count_distribution_dict:
                    sample_count_distribution_dict[n_bacterias] = 1
                else:
                    sample_count_distribution_dict[n_bacterias] += 1
        if bacteria_count_distribution:
            for i in range(self.n_bacterias):
                n_samples = np.sum(self.taxonomic_table[:, i] > 0)
                if n_samples not in bacteria_count_distribution_dict:
                    bacteria_count_distribution_dict[n_samples] = 1
                else:
                    bacteria_count_distribution_dict[n_samples] += 1
        return {"sample_count_distribution": sample_count_distribution_dict, "bacteria_count_distribution": bacteria_count_distribution_dict}

    def query_by_index(self, index: int, return_abundance: bool, return_metadata: bool):
        """
        query for a single sample by index
        query the HMC dataset using an integer index, the index follows the order
        in the taxonomic table
        returns a dictionary with the followings keys:
        - abundance: np array
        - metadata: dict
        """
        abundance = None
        metadata = None
        if return_abundance:
            abundance = self.taxonomic_table[index]

        if return_metadata:
            sample_name = self.sample_list.iloc[index]
            # load this sample's metadata
            sample_metadata = self.sample_metadata.loc[sample_name]
            # load this sample's project metadata
            project_name = sample_metadata["project"]
            project_metadata = self.project_metadata.loc[project_name]
            # load this sample's tags
            tags_index = self.tags_index_dict.get(sample_name.item(), [])
            tags = self.tags.loc[tags_index].drop(columns=["project", "srr", "srs"]).set_index("tag")

            sample_metadata_dict = sample_metadata.to_dict(orient="list")
            project_metadata_dict = project_metadata.to_dict(orient="list")
            tags_dict = tags["value"].to_dict()
            metadata = {**sample_metadata_dict, **project_metadata_dict}
            metadata["tags"] = tags_dict
        return {"abundance": abundance, "metadata": metadata}

    def generate_views(self, view_list: list):
        """
        generate views of the dataset
        returns a dictionary with the following keys:
        - every key in view_list
        """
        views = {}
        for view in view_list:
            if view == "study":
                # the dict for study corresponds to a list of studies
                studies = self.project_metadata.index.unique()
            if view == "num_sample_in_study":
                # the dict for num_sample_in_study corresponds to a dictionary with keys being the study and values being the number of samples
                num_sample_in_study = self.project_metadata.groupby("study").size().to_dict()
            if view == "number of tags":
                # the dict for number of tags corresponds to the number of unique tags
                num_tags = len(self.tags["tag"].unique())

    def get_num_samples_per_study(self, study: str):
        """
        returns a list of samples in a study
        """
        samples = self.sample_metadata[self.sample_metadata["project"] == study]
        return samples.shape[0]

    def get_tags_per_study(self):
        """
        returns a dictionary with keys being the study and values being the number of tags
        """
        tags_per_study = self.tags.groupby("project").unique().to_dict()
        return tags_per_study

    def temp(self):
        ret = self.tags.groupby('project')['tag'].unique().apply(list).reset_index(name='tags list')
        ret["tag_count"] = ret["tags list"].apply(len)
        ret["num_samples"] = ret["project"].apply(self.get_num_samples_per_study)
        return ret

    
    


# global paths for hmc dataset
taxonomic_table_path = "dataset/hmc/taxonomic_table.csv"
sample_metadata_path = "dataset/hmc/sample_metadata.tsv"
project_metadata_path = "dataset/hmc/projects.csv"
tags_path = "dataset/hmc/tags.tsv"

def load_dataset(filename):
    taxonomic_table_df = pd.read_csv(filename, nrows=5, index_col=0)
    return taxonomic_table_df

def load_metadata(filename):
    metadata_df = pd.read_csv(filename, sep='\t')
    return metadata_df

def test_hmc_class():
    hmc = HMC(taxonomic_table_path, project_metadata_path, sample_metadata_path, tags_path)
    # print(hmc.query_by_index(23, return_abundance=False, return_metadata=True))
    # print(hmc.generate_views(["study", "num_sample_in_study"]))
    temp = hmc.temp()
    print(temp.head())
    print(temp.shape)
    print(temp["tag_count"].sum())
    print(temp["num_samples"].sum())
    temp.to_csv("view_project_tags_ntags_nsamples.csv")

if __name__ == "__main__":
    print("starting script")
    # df = load_dataset(taxonomic_table_path)
    # df = load_metadata(sample_metadata_path)
    # df = pd.read_csv(project_metadata_path)
    # df1 = pd.read_csv(taxonomic_table_path, nrows=5, index_col=0)
    # df2 = pd.read_csv(taxonomic_table_path, usecols=[4680, 4681])
    # print("loaded table")
    # print(df1.shape)
    # print(df1.head())
    # print(df2.shape)
    # print(df2.head())
    test_hmc_class()
    # df = pd.read_csv(tags_path, sep='\t')
    # df["table_index"] = df["project"] + "_" + df["srr"]
    # df = df.set_index("table_index")
    # df = df.drop(columns=["project", "srr", "srs"])
    # index_dict = df.groupby("table_index").apply(lambda x: list(x.index)).to_dict()

    # print(df.head())
    # print(df.shape)
    # unique_entries = df["table_index"].unique()
    # unique_tags = df["tag"].unique()
    # print(len(unique_entries))
    # print(len(unique_tags))
    # q = "PRJDB10485_DRR243847"
    # # index_dict = df.groupby("table_index").apply(lambda x: list(x.index)).to_dict()
    # indices = index_dict[q]
    # small_df = df.loc[indices].set_index("tag")
    # print(small_df.columns)
    # print(small_df["value"].to_dict())
    # print(df[df["table_index"] == q])

    # study, number of samples, number of tags, tags