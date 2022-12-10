import pandas as pd
import os
import datetime

SPLITS = ['train', 'valid']
TIMEZONES = {
    'train':{
        'start': datetime.datetime(2017,3,29,12,0,0),
        'end': datetime.datetime(2017,7,23,3,0,0)
    },
    'valid':{
        'start': datetime.datetime(2017,7,23,3,0,0),
        'end': datetime.datetime(2017,9,9,23,0,0)}
    }


def main():
    with pd.option_context('display.max_columns', None):
        data_path = os.path.join('data', 'Flotation_Plant_preprocessed.csv')
        df = pd.read_csv(data_path)
        # string to datetime
        df['datetime'] = pd.to_datetime(df.datetime)
        # drop date column
        df = df.drop('date', axis=1)
        # normalize feature
        df.iloc[:,0:-1] = df.iloc[:,0:-1].apply(lambda x: round((x-x.min())/ (x.max()-x.min()),3), axis=0)

        filter = {split: (df['datetime'] >= TIMEZONES[split]['start']) & 
            (df['datetime'] < TIMEZONES[split]['end']) &
            (df['datetime'].dt.second == 0) & 
            (df['datetime'].dt.minute == 0) for split in SPLITS}

        dfs = {split: df.loc[filter[split]] for split in SPLITS}

        saved_path = {split: os.path.join('data', f"{split}.csv") for split in SPLITS}
        
        for split in SPLITS:
            dfs[split] = dfs[split].drop('datetime', axis=1)
            dfs[split].to_csv(saved_path[split], index=False)
    print("****** Generate Train/Valid Data in ./data/ ******")
    print(f"{'Training':<10} Data: {len(dfs['train']):>8}")
    print(f"{'Validation':<10} Data: {len(dfs['valid']):>8}")

if __name__ == '__main__':
    main()