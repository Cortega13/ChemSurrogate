import os
import numpy as np
import pandas as pd



class CSVtoHDF5:
    """
    Reads the UCLChem CSV output files (1 csv per model) and compresses them into a single HDF5 file.
    """
    def __init__(self, work_path, data_folder_name="grid_folder", output_filename="uclchem_rawdata_v6.h5"):
        self.work_path = work_path
        self.data_folder_path = os.path.join(work_path, data_folder_name)
        self.h5_store_path = os.path.join(work_path, output_filename)
        self.batch_size = 4192
        self.h5_store = pd.HDFStore(self.h5_store_path)

    @staticmethod
    def rename_columns(columns):
        """Renames columns to remove problematic characters and renames them to be more readable."""
        name_mapping = {
            'radfield': 'Radfield',
            '@H2COH': '@H3CO',
            'H2COH': 'H3CO',
            'point': 'Model',
            'H2CSH+': 'H3CS+',
            'SISH+': 'HSIS+',
            'E-': 'E_minus',
            'HOSO+': 'HSO2+',
            'H2COH+': 'H3CO+',
            'OCSH+': 'HOCS+',
            '#H2COH': '#H3CO',
        }
        columns = [col.strip() for col in columns]
        columns = [name_mapping[col] if col in name_mapping else col for col in columns]
        columns = [col.replace('#', 'SURF_')
                  .replace('+', 'Plus')
                  .replace('@', 'BULK_') for col in columns]
        return columns
    

    def process_and_store_data(self):
        """Reads CSV files, processes them, and stores them in an HDF5 file."""
        files_list = os.listdir(self.data_folder_path)
        batch_data = []
        global_index = 0
        
        for i, file in enumerate(files_list):
            if i % 100 == 0:
                print(f"Currently on Model: {i}")
            
            file_path = os.path.join(self.data_folder_path, file)
            single_model_data = pd.read_csv(file_path)
            single_model_data["Model"] = i
            
            row_count = len(single_model_data)
            single_model_data["Index"] = range(global_index, global_index + row_count)
            global_index += row_count
            
            single_model_data.columns = self.rename_columns(single_model_data.columns)
            
            single_model_data = single_model_data.drop(columns=["zeta", "point", "dustTemp", "SURFACE", "BULK"], errors='ignore')
            single_model_data = single_model_data.astype(np.float32)
            single_model_data["Model"] = single_model_data["Model"].astype(int)
            single_model_data = single_model_data.drop(index=1, errors='ignore')
            
            batch_data.append(single_model_data)
            
            if (i + 1) % self.batch_size == 0 or (i + 1) == len(files_list):
                combined_data = pd.concat(batch_data)
                self.h5_store.append('models', combined_data, format='table')
                batch_data = []
        
        print("Raw Data Saving Completed")
        self.h5_store.close()


    def run(self):
        """Executes the full compression process."""
        self.process_and_store_data()


class DatasetCleaner:
    """
    Confirms that timesteps are consistently 1kyr and clips abundances and physical parameters to preferred ranges.
    """
    def __init__(self, config):
        self.config = config
        self.working_path = config.working_path
        self.raw_filename = config.raw_filename
        self.df = None
    
    def load_data(self):
        self.df = pd.read_hdf(os.path.join(self.working_path, self.raw_filename), "models", start=0, dtype=np.float32)
        self.df = self.df.astype(np.float32)
        self.df.reset_index(drop=True, inplace=True)
        self.df.sort_values(by=["Model", "Time"], inplace=True)
        if "Index" not in self.df.columns:
            self.df['Index'] = range(len(self.df))
        print("-=+=- Dataset Loaded -=+=-")
        print(f"Original Total Dataset Size: {len(self.df)}")
    
    def clip_data(self):
        self.df = self.df.clip(lower=self.config.lower_clipping_threshold)
        for param, (min_val, max_val) in self.config.physical_parameter_ranges.items():
            if param in self.df.columns:
                self.df = self.df[(self.df[param] > min_val) & (self.df[param] < max_val)]
        self.df.infer_objects(copy=False)
        print("-=+=- Dataset Clipped by Threshold and Physical Parameter Ranges -=+=-")
    
    @staticmethod
    def filter_constant_timesteps(df, timestep=1000):
        df['diffs'] = df['Time'].diff().fillna(timestep)
        df['is_new_group'] = df['diffs'] != timestep
        df['temp_group'] = df['is_new_group'].cumsum()
        
        group_sizes = df.groupby('temp_group').size()
        max_group = group_sizes.idxmax()
        
        group_indices = df[df['temp_group'] == max_group].index
        start_index = group_indices[0]
        end_index = group_indices[-1]
        filtered_df = df.loc[start_index:end_index].drop(columns=['diffs', 'is_new_group', 'temp_group'])
        return filtered_df
    
    def process_data(self):
        df_constant_dt = (
            self.df.groupby('Model', group_keys=False)
            .apply(lambda group: self.filter_constant_timesteps(group.assign(Model=group.name)))
            .reset_index(drop=True)
        )
                
        print(f"Total Dataset Size: {len(df_constant_dt)} | Percentage: {len(df_constant_dt) / len(self.df) * 100:.2f}%")
        
        df_constant_dt.reset_index(drop=True, inplace=True)
        
        self.save_data(df_constant_dt, f"{self.config.data_category}.h5")
    
    def save_data(self, df, filename):
        df.to_hdf(os.path.join(self.working_path, filename), key="models", mode="a")
        print(f"-=+=- Data Successfully Saved: {filename} -=+=-")
    
    def run(self):
        self.load_data()
        self.clip_data()
        self.process_data()
