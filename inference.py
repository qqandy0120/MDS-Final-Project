from dataset import FlotationDataset
from train import PuritiesPredModule
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import argparse
import pandas as pd
def main(args):
    val_dataset = FlotationDataset('valid')
    val_dataloader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=256)
    model = PuritiesPredModule.load_from_checkpoint(args.ckpt_dir)
    trainer = Trainer()
    predictions = [float(item) for sublist in trainer.predict(model, dataloaders=val_dataloader) for item in sublist]
    labels = [float(item) for sublist in [batch['label'] for batch in val_dataloader] for item in sublist]

    dict = {'label': labels, 'prediction': predictions}
    df = pd.DataFrame(dict)
    df.to_csv(f'{args.ckpt_dir.split(".")[0]}_output.csv', index=False)
    ax = df.plot.line()
    ax.figure.savefig(f'{args.ckpt_dir.split(".")[0]}_output.png')

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        required=True,
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = arg_parse()
    main(args)