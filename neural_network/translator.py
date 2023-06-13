import re
import string
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.layers import (Dense, Dropout, Embedding, Layer,
                                     LayerNormalization, MultiHeadAttention,
                                     TextVectorization)


class TransformerEncoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = Sequential(
            [Dense(dense_dim, activation="relu"), Dense(embed_dim)])
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask)
        proj_input = self.layernorm_1(inputs+attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)


class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = Embedding(
            input_dim=vocab_size, output_dim=embed_dim)
        self.position_embeddings = Embedding(
            input_dim=sequence_length, output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoder(Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.attention_2 = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = Sequential(
            [Dense(latent_dim, activation="relu"), Dense(embed_dim),])
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.layernorm_3 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=out_1, value=encoder_outputs, key=encoder_outputs, attention_mask=padding_mask)
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1),
                         tf.constant([1, 1], dtype=tf.int32)], axis=0,)
        return tf.tile(mask, mult)


def translate_sequence(sequence, source_vectorization, target_vectorization, model, target_index_lookup):
    tokenized_input_sentence = source_vectorization([sequence])
    decoded_sentence = "[start]"
    for i in range(SEQUENCE_LENGTH):
        tokenized_target_sentence = target_vectorization([decoded_sentence])[
            :, :-1]
        predictions = model(
            [tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = target_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence


def translate(test_pairs, source_vectorization, target_vectorization, model):
    original = [pair[0] for pair in test_pairs]
    target_vocab = target_vectorization.get_vocabulary()
    target_index_lookup = dict(zip(range(len(target_vocab)), target_vocab))
    translated = []
    for i, sequence in enumerate(original):
        if i % 100 == 0:
            print(f'Translation {i} out of {len(original)}')
        translated.append(translate_sequence(
            sequence, source_vectorization, target_vectorization, model, target_index_lookup))
    return translated


def build_translation_pairs(source_file, target_file):
    source_text = load_from_file(source_file)
    target_text = load_from_file(target_file)
    text_pairs = [(source_text[i], f'[start] {target_text[i]} [end]') for i in range(
        len(target_text))]
    return text_pairs


def load_from_file(file_path):
    text = None
    with open(file_path) as f:
        text = f.read().split('\n')
    return text


def format_dataset(source, target):
    source = source_train_vectorization(source)
    target = target_train_vectorization(target)
    return ({"encoder_inputs": source, "decoder_inputs": target[:, :-1], }, target[:, 1:])


def build_dataset(pairs):
    batch_size = 64

    source_texts, target_texts = zip(*pairs)
    source_texts = list(source_texts)
    target_texts = list(target_texts)

    dataset = tf.data.Dataset.from_tensor_slices((source_texts, target_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()


def vectorize(pairs):

    source_vectorization = TextVectorization(max_tokens=VOCAB_SIZE,
                                             output_mode="int",
                                             output_sequence_length=SEQUENCE_LENGTH)
    target_vectorization = TextVectorization(max_tokens=VOCAB_SIZE,
                                             output_mode="int",
                                             output_sequence_length=SEQUENCE_LENGTH + 1,
                                             standardize=custom_standardization)
    source_texts = [pair[0] for pair in pairs]
    target_texts = [pair[1] for pair in pairs]
    source_vectorization.adapt(source_texts)
    target_vectorization.adapt(target_texts)
    return source_vectorization, target_vectorization


def custom_standardization(input_string):
    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


def get_model():
    encoder_inputs = Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE,
                            EMBED_DIM)(encoder_inputs)
    encoder_outputs = TransformerEncoder(EMBED_DIM, LATENT_DIM, NUM_HEADS)(x)

    decoder_inputs = Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = Input(
        shape=(None, EMBED_DIM), name="decoder_state_inputs")
    x = PositionalEmbedding(SEQUENCE_LENGTH, VOCAB_SIZE,
                            EMBED_DIM)(decoder_inputs)
    x = TransformerDecoder(EMBED_DIM, LATENT_DIM,
                           NUM_HEADS)(x, encoded_seq_inputs)
    x = Dropout(0.5)(x)
    decoder_outputs = Dense(VOCAB_SIZE, activation="softmax")(x)
    decoder = Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    decoder_outputs = decoder([decoder_inputs, encoder_outputs])
    model = Model([encoder_inputs, decoder_inputs],
                  decoder_outputs, name="transformer")
    model.summary()
    model.compile(
        "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


if __name__ == '__main__':
    datasets = {'raramuri-spanish': 'tar', 'shipibo_konibo-spanish': 'shp',
                'ashaninka-spanish': 'cni', 'wixarika-spanish': 'hch', 'aymara-spanish': 'aym', 'bribri-spanish': 'bzd', 'guarani-spanish': 'gn', 'hñähñu-spanish': 'oto', 'nahuatl-spanish': 'nah',
                'quechua-spanish': 'quy'}

    datasets = {'hñähñu-spanish': 'oto', 'nahuatl-spanish': 'nah',
                'quechua-spanish': 'quy'}

    VOCAB_SIZE = 15000
    SEQUENCE_LENGTH = 20
    EMBED_DIM = 256
    LATENT_DIM = 2048
    NUM_HEADS = 8
    EPOCHS = 20

    for k, v in datasets.items():
        model_path = f'./models/models/all-spanish/baseline_{EPOCHS}_epoch_model.h5'
        target_train_path = f'../data/{k}/train.es'
        source_train_path = f'../data/{k}/train.{v}'
        target_test_path = f'../data/{k}/test.es'
        source_test_path = f'../data/{k}/test.{v}'

        train_pairs = build_translation_pairs(
            source_train_path, target_train_path)
        test_pairs = build_translation_pairs(
            source_test_path, target_test_path)
        source_train_vectorization, target_train_vectorization = vectorize(
            train_pairs)
        train_dataset = build_dataset(train_pairs)
        test_dataset = build_dataset(test_pairs)
        try:
            model = get_model()
            model.load_weights(model_path)
        except Exception as e:
            print(e)
            continue
        result = translate(test_pairs, source_train_vectorization,
                           target_train_vectorization, model)
        Path(f"./results/{k}").mkdir(parents=True, exist_ok=True)
        with open(f'./results/{k}/baseline_{EPOCHS}_all_epoch_result', 'w+') as f:
            for line in result:
                out = line.replace('[start]', '').replace('[end]', '')
                f.write(out+'\n')
