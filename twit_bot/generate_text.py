import sys

import click
from keras.models import load_model
from rnaseq_lib.utils import rexpando
from train_rnn import generate_seed, str_to_vec, char_maps, sample


@click.command()
@click.option('--model', required=1, help='Path to Keras model (HDF5)')
@click.option('--text', required=1, help='Path to text used to generate model (to rebuild char sets)')
@click.option('--maxlen', default=40, help='Max Length used to build model')
@click.option('--temp', default=0.0, help='Temperature - affects sampling from softmax probability vector')
@click.option('--num-chars', default=400, help='Number of characters to generate')
@click.option('--seed', default=None, help='If provided, will use instead of a random seed from text')
def main(model, text, maxlen, temp, num_chars, seed):
    """
    Generates text given a model and a random seed generated from a text

    Example: generate-text --model=model.hdf5 --text=data/training-text.txt

    Temperature (--temp) affects multinomial sampling of the prediction vector. Higher values can help break
    the model of infinite loops by introducing entropy into the sequence generation. Typical range [0.0 - 1.5].
    """
    click.echo('-- Settings -- ')
    click.echo(''.join(['{}\t{}\n'.format(x, y) for x, y in sorted(locals().iteritems())]))

    # Store options
    opts = rexpando(locals())

    # Read in text
    click.echo('Reading in text')
    opts.text = open(text, 'r').read()

    # If a seed is not provided, generate one from text
    if not seed:
        click.echo('Generating seed from text')
        seed = generate_seed(opts)
    else:
        opts.maxlen = len(seed)

    # Generate character indices from text
    opts.len_chars, opts.char_indices, opts.indices_char = char_maps(opts.text)

    # Load model
    model = load_model(model)

    # Generate text
    click.clear()
    click.echo('\n-- Text Generation --\nSeed: {}'.format(seed))
    sentence = '' + seed
    sys.stdout.write(seed)
    for i in xrange(num_chars):
        # Convert text seed to input
        x_pred = str_to_vec(sentence, opts)

        # Make prediction with model
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temp)
        next_char = opts.indices_char[next_index]

        # Generate text
        seed += next_char
        sentence = sentence[1:] + next_char

        # Output text to terminal as it is generated
        sys.stdout.write(next_char)
        sys.stdout.flush()


if __name__ == '__main__':
    main()
