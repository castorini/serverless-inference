import boto3
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

# change this to use other models
from sm_cnn.model import SMModel


client = boto3.client('dynamodb')
table_name = 'word2vec50d'

def build_matrix(words, lookup):
    matrix = None
    for word in words:
        if word in lookup:
            vec_raw = lookup[word]
            vec = np.array([float(f['N']) for f in vec_raw])
        else:
            # random vector
            vec = np.random.rand(50)

        vec = vec.reshape(1, 1, vec.shape[0])
        if matrix is None:
            matrix = vec
        else:
            matrix = np.append(matrix, vec, axis=1)

    return matrix


def sentence_to_matrix(sentence1, sentence2):
    """
    Get word vectors for word in sentence and build a matrix with the vectors.
    """
    words1 = sentence1.split(' ')
    words2 = sentence2.split(' ')

    # request cannot contain duplicate keys. remove duplicates
    words_no_dup = list(set(words1) | set(words2))
    read_batch_size = 100
    batches = [words_no_dup[i:i+read_batch_size] for i in range(0,len(words_no_dup),read_batch_size)]
    wordvec_a = []
    for batch in batches:
        request = [{'word':{'S':word}} for word in batch]

        response = client.batch_get_item(
            RequestItems = {
                table_name: {
                    'Keys': request
                }
            }
        )

        wordvec_a = wordvec_a + [(d['word']['S'], d['vector']['L']) for d in response['Responses'][table_name]]

    lookup = dict(wordvec_a)
    print('Found words:', list(lookup.keys()))

    matrix1 = build_matrix(words1, lookup)
    matrix2 = build_matrix(words2, lookup)
    return matrix1, matrix2


def handler(event, context):
    body = json.loads(event['body'])
    sentence1 = body['sent1']
    sentence2 = body['sent2']

    input_matrix1, input_matrix2 = sentence_to_matrix(sentence1, sentence2)

    # load and run model
    # you may need to modify this based on your model definition
    model = torch.load('static_best_model.pt')
    model.eval()
    torchIn1 = Variable(torch.from_numpy(input_matrix1.astype(np.float32)))
    torchIn2 = Variable(torch.from_numpy(input_matrix2.astype(np.float32)))
    output = model(torchIn1, torchIn2)
    score = output.data.numpy().tolist()

    # return result
    result = {
        'sent1': sentence1,
        'sent2': sentence2,
        'score': score
    }
    return {
        'statusCode': 200,
        'headers': { 'Content-Type': 'application/json' },
        'body': json.dumps(result)
    }


if __name__ == '__main__':
    event = {
        'body': json.dumps({
            'sent1': 'how are glacier caves formed',
            'sent2': 'the ice facade is approximately 60 m high'
        })
    }
    print(handler(event, None))

