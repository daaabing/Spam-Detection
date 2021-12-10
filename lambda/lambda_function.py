import json
import urllib.parse
import boto3
from datetime import datetime
import numpy as np
from hashlib import md5
import string
import sys
import os

s3 = boto3.client('s3')
ses = boto3.client('ses')
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    email = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    try:
        response = s3.get_object(Bucket=bucket, Key=email)
        raw = response['Body'].read().decode("utf-8").split("\n")
        print("response", raw)
        SENDER = ""
        EMAIL_RECEIVE_DATE = ""
        EMAIL_SUBJECT = ""
        EMAIL_BODY = ""
        boundary = ""
        i = 0
        # parse raw email message into parts and ignore other information
        while i < len(raw):
            raw[i] = raw[i].rstrip()
            if raw[i].startswith("From: "):
                SENDER = raw[i].split("<")[-1][:-1]
            elif raw[i].startswith("Date:"):
                date_time_str = raw[i].replace("Date: ", "")
                EMAIL_RECEIVE_DATE = datetime.strptime(date_time_str[:-6], '%a, %d %b %Y %H:%M:%S')
            elif raw[i].startswith("Subject: "):
                EMAIL_SUBJECT = raw[i].replace("Subject: ", "")
                i += 2  # jumpy to first boundary
                l = raw[i]
                if l.startswith("MIME-Version"):
                    i += 1
                    l = raw[i]
                print("L", l)
                boundary = "--" + l[10+l.index("boundary"):-2]
                print(boundary)
                while not raw[i].startswith(boundary): i += 1
                i += 1
                while raw[i].startswith("Content-"):
                    i += 1  # skip content type before body
                print(raw[i])
            elif boundary != "":
                if raw[i].startswith(boundary): # body is surround by two boundary string
                    break
                EMAIL_BODY += raw[i]
            i += 1
        print("sending %s to the endpoint" % EMAIL_BODY)
        one_hot_test_messages = one_hot_encode([EMAIL_BODY], 9013)
        encoded_test_messages = vectorize_sequences(one_hot_test_messages, 9013)

        payload = json.dumps(encoded_test_messages.tolist())
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            Body=payload,
            Accept='application/json',
        )
        res = json.loads(response['Body'].read().decode())
        print(res)
        CLASSIFICATION = "SPAM" if res["predicted_label"][0][0] == 1 else "NON-SPAM"
        CLASSIFICATION_CONFIDENCE_SCORE = res["predicted_probability"][0][0] * 100
        print(CLASSIFICATION, CLASSIFICATION_CONFIDENCE_SCORE)
        text = f"Hi {SENDER}! We received your email sent at {EMAIL_RECEIVE_DATE} with the subject {EMAIL_SUBJECT}.\n\nHere is a 240 character sample of the email body:\n"
        text += f"{EMAIL_BODY}\nThe email was categorized as {CLASSIFICATION} with a "
        text += "%.2f%% confidence." % CLASSIFICATION_CONFIDENCE_SCORE
        print(text)
        
        response = ses.send_email(
            Destination={
                'ToAddresses': [
                    SENDER,
                ],
            },
            Message={
                'Body': {
                    'Text': {
                        'Charset': "UTF-8",
                        'Data': text,
                    },
                },
                'Subject': {
                    'Charset': "UTF-8",
                    'Data': "Result",
                },
            },
            Source="hw@hw3-email-hw.co",
        )
        
        return text
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(email, bucket))
        raise e


if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans

    
def vectorize_sequences(sequences, vocabulary_length):
    results = np.zeros((len(sequences), vocabulary_length))
    for i, sequence in enumerate(sequences):
      results[i, sequence] = 1. 
    return results

def one_hot_encode(messages, vocabulary_length):
    data = []
    for msg in messages:
        temp = one_hot(msg, vocabulary_length)
        data.append(temp)
    return data

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to convert the input to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,):
        if isinstance(text, unicode):
            translate_map = dict((ord(c), unicode(split)) for c in filters)
            text = text.translate(translate_map)
        elif len(split) == 1:
            translate_map = maketrans(filters, split * len(filters))
            text = text.translate(translate_map)
        else:
            for c in filters:
                text = text.replace(c, split)
    else:
        translate_dict = dict((c, split) for c in filters)
        translate_map = maketrans(translate_dict)
        text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]

def one_hot(text, n,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '):
    """One-hot encodes a text into a list of word indexes of size n.
    This is a wrapper to the `hashing_trick` function using `hash` as the
    hashing function; unicity of word to index mapping non-guaranteed.
    # Arguments
        text: Input text (string).
        n: int. Size of vocabulary.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        List of integers in [1, n]. Each integer encodes a word
        (unicity non-guaranteed).
    """
    return hashing_trick(text, n,
                         hash_function='md5',
                         filters=filters,
                         lower=lower,
                         split=split)


def hashing_trick(text, n,
                  hash_function=None,
                  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                  lower=True,
                  split=' '):
    """Converts a text to a sequence of indexes in a fixed-size hashing space.
    # Arguments
        text: Input text (string).
        n: Dimension of the hashing space.
        hash_function: defaults to python `hash` function, can be 'md5' or
            any function that takes in input a string and returns a int.
            Note that 'hash' is not a stable hashing function, so
            it is not consistent across different runs, while 'md5'
            is a stable hashing function.
        filters: list (or concatenation) of characters to filter out, such as
            punctuation. Default: `!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n`,
            includes basic punctuation, tabs, and newlines.
        lower: boolean. Whether to set the text to lowercase.
        split: str. Separator for word splitting.
    # Returns
        A list of integer word indices (unicity non-guaranteed).
    `0` is a reserved index that won't be assigned to any word.
    Two or more words may be assigned to the same index, due to possible
    collisions by the hashing function.
    The [probability](
        https://en.wikipedia.org/wiki/Birthday_problem#Probability_table)
    of a collision is in relation to the dimension of the hashing space and
    the number of distinct objects.
    """
    if hash_function is None:
        hash_function = hash
    elif hash_function == 'md5':
        hash_function = lambda w: int(md5(w.encode()).hexdigest(), 16)

    seq = text_to_word_sequence(text,
                                filters=filters,
                                lower=lower,
                                split=split)
    return [int(hash_function(w) % (n - 1) + 1) for w in seq]