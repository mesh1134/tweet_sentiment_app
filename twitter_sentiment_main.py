import re

import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.callbacks import EarlyStopping

# Constants
MAX_WORDS = 20000
MAX_LEN = 40
EMBEDDING_DIM = 100
NUM_CLASSES = 3
EPOCHS = 10


# --- THE CLEANING FUNCTION ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    print("Loading Data...")
    df = pd.read_csv('Twitter_Data.csv')
    df = df.dropna(subset=['text', 'sentiment'])

    # 1. APPLY CLEANING (Critical Step)
    print("Cleaning Texts...")
    df['text'] = df['text'].apply(clean_text)

    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment_code'] = df['sentiment'].map(label_map)

    x_train, x_test, y_train, y_test = train_test_split(
        df['text'], df['sentiment_code'],
        test_size=0.2, random_state=42, stratify=df['sentiment_code']
    )

    # 2. Tokenize
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train)

    x_train_pad = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=MAX_LEN, padding='post')
    x_test_pad = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=MAX_LEN, padding='post')

    # 3. Model
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.2),
        Bidirectional(LSTM(128, dropout=0.2)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # 4. Train
    model.fit(x_train_pad, y_train,
              epochs=EPOCHS,
              batch_size=64,
              validation_data=(x_test_pad, y_test),
              callbacks=[early_stop]
              )

    # 5. Save
    model.save('sentiment_model.keras')
    dump(tokenizer, 'tokenizer.joblib')
    print("Training Complete. Files Saved.")


if __name__ == "__main__":
    main()
