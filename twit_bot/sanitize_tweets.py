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
    chars_to_remove = {'"', '%', '(', ')', '*', '+', '/', ';',
                       '<', '=', '>', '[', '\\', ']', '_', '`', '{', '|', '}', '~'}

    # Sort dataframe chronologically for contextual continuity
    df['created'] = pd.to_datetime(df.created)
    df = df.sort_values('created')

    # Iterate over all tweets (ignoring retweets)
    click.echo('Processing {}'.format(tweets))
    cleaned_tweets = []
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
            # if tweet[-1] not in ['.', '!', '?']:
            #   tweet = tweet + '.'

            # Add to text (by redefining with +=)
            cleaned_tweets.append(tweet)

    # Output
    if not os.path.exists('data'):
        os.mkdir('data')

    # Combine tweets into a single text
    with open('data/combined-tweets.txt', 'w') as f:
        f.write(' '.join(cleaned_tweets))

    # Tweets on separate lines
    with open('data/separated-tweets.txt',  'w') as f:
        f.write('\n'.join(cleaned_tweets))
    click.echo('Output saved in `data`. {} tweets used to build text.'.format(len(cleaned_tweets)))


if __name__ == '__main__':
    sanitize_tweets()
