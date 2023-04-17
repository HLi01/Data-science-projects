import pandas as pd

class NJCleaner:

    def __init__(self, csv_path:str):
        df = pd.read_csv(csv_path)
        self.data = df

    def order_by_scheduled_time(self) -> pd.DataFrame:
        self.data = self.data.sort_values(by=['scheduled_time'])
        return self.data
    
    def drop_columns_and_nan(self) -> pd.DataFrame:
        self.data = self.data.drop(['from', 'to'], axis=1)
        self.data = self.data.dropna()
        return self.data

    def convert_date_to_day(self) -> pd.DataFrame:
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['day'] = self.data['date'].dt.day_name()
        self.data = self.data.drop(['date'], axis=1)
        return self.data
    
    def convert_scheduled_time_to_part_of_the_day(self) -> pd.DataFrame:
        df = pd.DatetimeIndex(self.data['scheduled_time']).hour
        self.data['part_of_the_day'] = pd.cut(df, bins=[-1, 3, 7, 11, 15, 19, 24], 
                                              labels=[ 'late_night', 'early_morning', 'morning', 'afternoon', 'evening', 'night']) 
        self.data.drop(columns=['scheduled_time'], inplace=True) 
        return self.data

    def convert_delay(self) -> pd.DataFrame:
        max = self.data['delay_minutes'].astype('float').max()
        self.data['delay'] = pd.cut(self.data['delay_minutes'].astype('float'), bins=[-1, 4.999, max], labels=[0, 1])
        return self.data

    
    def drop_unnecessary_columns(self) -> pd.DataFrame:
        self.data.drop(['train_id', 'actual_time', 'delay_minutes'], axis=1, inplace=True)
        return self.data
    
    def save_first_60k(self, path:str):
        cutted_df = self.data.head(60000)
        cutted_df.to_csv(path, index=False)

    def prep_df(self, path:str='data/NJ.csv'):
        self.order_by_scheduled_time()
        self.drop_columns_and_nan()
        self.convert_date_to_day()
        self.convert_scheduled_time_to_part_of_the_day()
        self.convert_delay()
        self.drop_unnecessary_columns()
        self.save_first_60k(path)