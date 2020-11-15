import pandas as pd

from bit_agent import BitAgent

price_data = pd.read_csv('bitcoin_price.csv', index_col=0, header=0)

trainer = BitAgent(price_data, model_path='./models/v1.pt',
                   logdir='./logdir/v1')


trainer.train(2000000)
