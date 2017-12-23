import os
import re

import click
import pandas as pd


@click.command()
@click.argument('tweets', required=1)
def sanitize_tweets(tweets):
    """
    Sanitizes individual tweets into a single text:
        - Sort tweets chronologically
        - Remove URLs
        - Fix HTML encoding errors
        - Lower-case characters
        - Remove non-ASCII / unimportant to limit char set
        - Fix errant spacing

    :param str tweets: Path to tweet TSV file. Columns: source/text/created/retweets/favorites/is_rtweet/id
    """
    click.echo('Reading in tweets')
    df = pd.read_csv(tweets, sep='\t', index_col=0)

    # Character set to omit from tweets
    chars_to_remove = {'"', '$', '%', "'", '(', ')', '*', '+', '/', ';',
                       '<', '=', '>', '[', '\\', ']', '_', '`', '{', '|', '}', '~'}

    # Sort dataframe chronologically for contextual continuity
    df['created'] = pd.to_datetime(df.created)
    df = df.sort_values('created')

    # Iterate over all tweets (ignoring retweets)
    click.echo('Processing {}'.format(tweets))
    tweet_counter = 0
    text = ''
    with click.progressbar(enumerate(df[df.is_retweet == False].text)) as selected_tweets:
        for i, tweet in selected_tweets:
            # Remove URLs
            tweet = re.sub(r'http\S+', '', tweet)

            # Fix ampersand HTML artifact
            tweet = tweet.replace('&amp;', '&')

            # lower() to reduce pool of possible characters (lower-case's strings)
            # the decode/encode step is to remove non-ascii characters like
            tweet = tweet.lower().decode("ascii", errors="ignore").encode()

            # Remove chars from our chars_to_remove set with list comprehension
            for x in chars_to_remove:
                tweet = tweet.replace(x, ' ')

            # If there's no tweet left (just a URL for example)
            if not tweet:
                continue

            # split and rejoin for odd spacing
            tweet = ' '.join(tweet.split())

            # Add period if no ending line or a space
            if not tweet[-1] in ['.', '!', '?']:
                tweet = tweet + '. '
            else:
                tweet += ' '

            # Add to text (by redefining with +=)
            text += tweet
            tweet_counter += 1

    # Output
    if not os.path.exists('data'):
        os.mkdir('data')

    output = 'data/sanitized-tweets.txt'
    with open(output, 'w') as f:
        f.write(text)
    click.echo('Output saved: {}\n{} tweets used to build text.'.format(output, tweet_counter))


if __name__ == '__main__':
    sanitize_tweets()
