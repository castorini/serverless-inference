"""
Upload word vectors to DynamoDB.
Original author: @ShawnLMP
Modified by: @tuzhucheng
"""

from argparse import ArgumentParser

import boto3
import numpy
import torch
from torch.autograd import Variable

dynamodb = boto3.client('dynamodb')


def word_to_put_req(word_vectors_map, word):
    """
    Translate a word to a PUT request to be sent to DynamoDB
    """
    return {
        'PutRequest': {
            'Item': {
                'word': {
                    'S': word
                },
                'vector': {
                    'L': [{'N': str(n)} for n in word_vectors_map[word]]
                }
            }
        }
    }


def sublist(l):
    """
    Breaks a long list into small sublists
    """
    return [l[i:i+25] for i in range(0,len(l), 25)] # 25 is the max request we can sent to dynamo in one batch


def get_word_vectors_map(words, filename):
    """
    Get mapping from words to word vectors. Words are selected from words in words.
    If words is empty, get all word vectors from the word vectors textfile specified by filename.
    """
    word_set = set(words)
    word_vectors_map = {}
    header_found = False
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split(' ')
            if len(parts) < 50 and not header_found:
                print('Ignoring header row')
                header_found = True
                continue

            if len(word_set) == 0 or parts[0] in word_set:
                vec = list(map(float, parts[1:]))
                word_vectors_map[parts[0]] = vec

    return word_vectors_map


def put_words(word_vectors_map, table_name):
    """
    Upload word vectors to DynamoDB
    """
    batches = sublist(list(word_vectors_map.keys()))

    for i, batch in enumerate(batches):
        request = [word_to_put_req(word_vectors_map, word) for word in batch]
        response = dynamodb.batch_write_item(
            RequestItems = {
                table_name: request
            }
        )

        if i % 10 == 0:
            print('Finished batch', i+1)


if __name__ == '__main__':
    parser = ArgumentParser(description='Upload word vectors in GloVe / word2vec txt format to DynamoDB')
    parser.add_argument('word_vectors_file', help='word vectors path')
    parser.add_argument('table_name', help='target table name in Dynamo')
    parser.add_argument('--word_list', help='word list for filtering words to upload')
    args = parser.parse_args()

    words = []
    if args.word_list is not None:
        with open(args.word_list) as f:
            for line in f:
                words.append(line.rstrip())

    word_vectors_map = get_word_vectors_map(words, args.word_vectors_file)
    print('Number of word vectors to upload: ', len(word_vectors_map))
    put_words(word_vectors_map, args.table_name)
