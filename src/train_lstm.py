from utils import * 
import os 
import tensorflow as tf
import tensorflow_addons as tfa

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

BUFFER_SIZE = 1000
BATCH_SIZE = 100
VOCAB_SIZE = 10000


def run(df, epochs, N_CLASS, metric, fold):
    # separate train and test
    train_dataset = df[df["kfold"] != fold].reset_index(drop=True)
    test_dataset = df[df["kfold"] == fold].reset_index(drop=True)
    
    x_train = train_dataset.drop(["kfold", "category"], axis=1).values
    x_test = test_dataset.drop(["kfold", "category"], axis=1).values

    # one hot the target variable
    y_train = tf.one_hot(train_dataset.category.values, N_CLASS)
    y_test = tf.one_hot(test_dataset.category.values, N_CLASS)
    
    # make tf compatible dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Start model 
    encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))
    
    # embedding + 1 * bidirectional LSTM(64) + Dense(64) + Dense output
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=2000,
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512)), # return_sequences=True)),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256)),
        # tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(N_CLASS)
    ])
    
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=[metric])
    
    history = model.fit(train_dataset, epochs=epochs,
                    validation_data=test_dataset) # , validation_steps=30)

def main():
    # import data
    df = pd.read_csv(f"/content/drive/MyDrive/MIDAS/data/text.csv")
    df.drop(["image"], axis=1, inplace=True)
    # Lable Encoding the target variable
    enc = preprocessing.LabelEncoder()
    df["category"] = enc.fit_transform(df["category"])

    # defining the metric
    N_CLASS = len(df["category"].unique())
    metric = tfa.metrics.F1Score(num_classes=N_CLASS, average="micro")

    print("N_CLASS:", N_CLASS)

    for fold in range(5):
        run(df=df, epochs=10, N_CLASS=N_CLASS, metric=metric, fold=fold)

if __name__ == "__main__":
    main()