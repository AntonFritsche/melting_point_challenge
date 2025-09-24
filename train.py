from model import lstm_embedding, lstm_stacked

import pandas as pd
import tensorflow as tf


df = pd.read_csv('melting-point/train.csv')
features = [c for c in df.columns if c.startswith("Group")]
targetCols = "Tm"
embedding_size = 256

X = df[features].values
y = df[targetCols].values

model_embedding = lstm_embedding(len(features), embedding_size, 1)
model_stacked = lstm_stacked(len(features))

model_embedding.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_stacked.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_embedding.fit(X, y, epochs=10, batch_size=256, verbose=1)
model_stacked.fit(X, y, epochs=10, batch_size=256, verbose=1)
