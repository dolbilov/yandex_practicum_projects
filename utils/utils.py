import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from numpy.typing import ArrayLike
from prettytable import PrettyTable


class Utils:
    @staticmethod
    def print_table_about_nans(dataframes, dfs_names) -> None:
        all_columns = set()
        for df in dataframes:
            all_columns |= set(df.columns)
            
        table = PrettyTable()
        table.field_names = ['Feature'] + [f'{name} missing count (%)' for name in dfs_names]
        
        for column in all_columns:
            row = [column]
            
            for df in dataframes:
                if column not in df:
                    row.append('NA')
                else:
                    na_count = df[column].isna().sum()
                    na_percent = round(na_count / len(df) * 100, 1)
                    row.append(f'{na_count} ({na_percent}%)')
            
            table.add_row(row)
        
        print(table)
    
    @staticmethod
    def apply_min_max_scaler(train: pd.DataFrame,
                             test: pd.DataFrame,
                             cols: ArrayLike
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
        sc = MinMaxScaler()
        
        combined = pd.concat([train, test], axis=0)
        combined[cols] = sc.fit_transform(combined[cols])
        
        return combined.iloc[:len(train)], combined.iloc[len(train):]
    
    @staticmethod
    def apply_one_hot_encoding(train: pd.DataFrame,
                               test: pd.DataFrame,
                               cols: ArrayLike,
                               drop_least_category: bool = False
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ohe = OneHotEncoder()
        
        combined = pd.concat([train, test], axis=0)
        encoded_data = pd.DataFrame(
            data=ohe.fit_transform(combined[cols]).toarray(),
            index = combined.index,
            columns=ohe.get_feature_names_out(cols)
        )
        combined = combined.drop(cols, axis=1)
        combined = combined.join(encoded_data)
        
        return combined.iloc[:len(train)], combined.iloc[len(train):]
    
    @staticmethod
    def apply_label_encoder(train: pd.DataFrame,
                            test: pd.DataFrame,
                            cols: ArrayLike
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        
        combined = pd.concat([train, test], axis=0)
        
        for col in cols:
            enc = LabelEncoder()
            combined[col] = enc.fit_transform(combined[col])
            
        return combined.iloc[:len(train)], combined.iloc[len(train):]
