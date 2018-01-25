from __future__ import print_function

import os
import timeit

import click
import numpy as np
from keras.callbacks import CSVLogger
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GRU
from keras.layers import LSTM
from keras.models import Sequential
from rnaseq_lib.utils import mkdir_p
from rnaseq_lib.utils import rexpando


def tf_gpu_growth():
    """
    Establishes a tensorflow session with smarter GPU memory allocation
    """
    import tensorflow as tf
    from keras import backend as K
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    K.set_session(session)


def char_maps(text):
    """
    Create character mappings from text (and return len_chars)

    :param str text: Input text to create mappings from
    :return: len_chars, c->i mapping, i->c mapping
    :rtype: tuple(int, dict(str, int), dict(int, str))
    """
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return len(chars), char_indices, indices_char


def create_training_set(opts):
    """
    Break up text into sentences and next_chars (X and y respectively) then vectorize

    Use newlines to separate source material for training examples.

    :param Expando opts: Meta-object to store options and intermediate objects
    :return: Training pair
    :rtype: tuple(np.array, np.array)
    """
    # Break up text into sentences and next_chars
    sentences = []
    next_chars = []
    # for sample in opts.text.split('\n'):
    #     if len(sample) > opts.maxlen + opts.stride:
    for i in range(0, len(opts.text) - opts.maxlen, opts.stride):
        sentences.append(opts.text[i: i + opts.maxlen])
        next_chars.append(opts.text[i + opts.maxlen])

    # Create X and y tensors from training set
    x = np.zeros((len(sentences), opts.maxlen, opts.len_chars), dtype=np.bool)
    y = np.zeros((len(sentences), opts.len_chars), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            x[i, t, opts.char_indices[char]] = 1
        y[i, opts.char_indices[next_chars[i]]] = 1

    return x, y


def create_model(opts):
    """
    Creates RNN model

    :param Expando opts: Meta-object to store options and intermediate objects
    :return: RNN model
    :rtype: Sequential
    """
    # Set type of layer to use
    layer = GRU if opts.gru else LSTM

    # Create a sequential model
    model = Sequential()

    # Add RNN layers
    if opts.num_layers == 1:
        model.add(layer(opts.units, input_shape=(opts.maxlen, opts.len_chars)))
        if opts.dropout != 0:
            model.add(Dropout(opts.dropout))

    # Multiple RNN layers require a return_sequences=True to join together
    else:
        model.add(layer(opts.units, return_sequences=True,
                        input_shape=(opts.maxlen, opts.len_chars)))
        model.add(Dropout(opts.dropout))
        for i in xrange(opts.num_layers - 2):
            model.add(layer(opts.units, return_sequences=True))
            if opts.dropout != 0.0:
                model.add(Dropout(opts.dropout))

        # Add final RNN layer without `return_sequences=True` or model will fail during `.fit()`
        model.add(layer(opts.units))
        if opts.dropout != 0.0:
            model.add(Dropout(opts.dropout))

    # Add final Dense layer
    model.add(Dense(opts.len_chars, activation='softmax'))

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def str_to_vec(sentence, opts):
    """
    Convert string to vector format for model input

    :param str sentence: String to convert
    :param Expando opts: Meta-object to store options and intermediate objects
    :return: Vectorized input
    :rtype: np.array
    """
    x_pred = np.zeros((1, opts.maxlen, opts.len_chars))
    for t, char in enumerate(sentence):
        x_pred[0, t, opts.char_indices[char]] = 1.
    return x_pred


def generate_seed(opts):
    """
    Generates random seed from text

    :param Expando opts: Meta-object to store options and intermediate objects
    :return: Random seed
    :rtype: str
    """
    for line in opts.text.split('\n'):
        if len(line) > opts.maxlen:
            start_index = np.random.randint(0, len(line) - opts.maxlen - 1)
            return opts.text[start_index: start_index + opts.maxlen]


def sample(preds, temperature):
    """
    Taken from: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py#L63

    :param np.array preds: Prediction vector from model
    :param float temperature: Amount of variation in softmax probability - 0=no variation, >1 = lot of variation
    :return: Index of choice in character vector
    :rtype: int
    """
    # helper function to sample an index from a probability array
    temperature = float(temperature)
    if temperature > 0:
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    else:
        return np.argmax(preds)


def generate_text_from_seed(model, opts, num_chars=400):
    """
    Generates text from a random seed

    :param Sequential model: Model to use to generate characters
    :param Expando opts: Meta-object to store options and intermediate objects
    :param int num_chars: Number of characters to generate
    """
    seed = generate_seed(opts)
    echo('\n-- Text Generation --\nSeed: {}'.format(seed), filename=opts.log)
    sentence = '' + seed

    for temp in [0.0, 0.5, 1.0]:
        echo('Temp: {}'.format(temp), filename=opts.log)
        for i in xrange(num_chars):
            # Convert text seed to input
            x_pred = str_to_vec(sentence, opts)

            # Make prediction with model
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, temp)  # opts.indices_char[np.argmax(preds)]
            next_char = opts.indices_char[next_index]

            # Generate text
            seed += next_char
            sentence = sentence[1:] + next_char

        echo('Generated: {}'.format(seed), filename=opts.log)


def echo(message, filename):
    """
    Wrapper for click.echo to get both stdout and output to file

    :param str message:
    :param str filename:
    """
    click.echo(message)
    click.echo(message, file=open(filename, 'a'))


@click.command()
@click.argument('text', required=1)
@click.option('--maxlen', default=40, help='Length of input string for training')
@click.option('--stride', default=3, help='Stride window when creating training sets')
@click.option('--epochs', default=20, help='Number of epochs to train for')
@click.option('--batch-size', default=128, help='Batch size when training')
@click.option('--num-layers', default=1, help='Number of LSTM/GRU layers to add')
@click.option('--units', default=256, help='Number of memory units in RNN layer')
@click.option('--dropout', default=0.2, help='Dropout value added after RNN layers (between 0 and 1)')
@click.option('--gru', is_flag=True, help='Use GRU layers instead of LSTM layers')
@click.option('--gpu-growth', is_flag=True, help='Sets Tensorflow to only use as much memory as needed.')
@click.option('--weights-per-epoch', is_flag=True, help='Outputs weights every epoch')
def train(text, maxlen, stride, epochs, batch_size, num_layers, units, dropout, gru, gpu_growth, weights_per_epoch):
    """
    Train a recurrent neural network (RNN) on a provided TEXT

    Example: `train-rnn data/sanitized-tweets.txt`
    """
    click.clear()

    # Set GPU
    if gpu_growth:
        tf_gpu_growth()

    # Create directories for model
    text_name = os.path.splitext(os.path.basename(text))[0]
    run_name = '{maxlen}-{stride}-{epochs}-{batch_size}-{num_layers}-{units}-{dropout}-{gru}'.format(**locals())
    out_dir = os.path.join('models', text_name, run_name)
    model_out = os.path.join(out_dir, 'model.hdf5')
    if os.path.exists(model_out):
        click.echo('Model already exists: {}, exiting'.format(model_out))
        exit()
    mkdir_p(out_dir)

    # Print options
    log = os.path.join(out_dir, 'log.txt')
    echo('-- Network Settings -- ', filename=log)
    echo(''.join(['{}\t{}\n'.format(x, y) for x, y in sorted(locals().iteritems())]), filename=log)

    # Object to hold options and settings
    opts = rexpando(locals())

    # Read input text
    opts.text = open(text, 'r').read()

    # Create index / character mappings
    opts.len_chars, opts.char_indices, opts.indices_char = char_maps(opts.text)

    # Create training set
    x, y = create_training_set(opts)
    echo('Created {} training samples from text'.format(len(y)), filename=log)

    # Create model
    model = create_model(opts)

    # Define logger for training callbacks
    logger = CSVLogger(os.path.join(out_dir, 'model-history.tsv'), append=True, separator='\t')

    # Train model
    start = timeit.default_timer()
    for i in xrange(epochs):
        echo('\nEpoch: {}'.format(i + 1), filename=opts.log)

        # Fit model
        model.fit(x, y, batch_size=batch_size, epochs=1, callbacks=[logger])

        # Generate text
        generate_text_from_seed(model, opts)

        # Output weights
        if weights_per_epoch:
            epoch_out = os.path.join(out_dir, '{}-model.hdf5'.format(i))
            model.save(epoch_out)
            click.echo('Saving weights for epoch: {}'.format(i))

    # Calculate runtime
    runtime = timeit.default_timer() - start
    echo('Training runtime: {}'.format(runtime), filename=opts.log)

    # Save model
    model.save(model_out)
    click.echo('Final model saved: {}'.format(model_out))


if __name__ == '__main__':
    train()
