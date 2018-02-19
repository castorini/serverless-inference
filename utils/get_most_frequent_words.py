import csv
import sys
from argparse import ArgumentParser

def get_frequent_words(frequency_data_file , num_of_words):
    """
    Get the most frequent `num_of_words` words from `frequency_data_file` data set file.
    Please download the word frequency dataset here:
        https://www.kaggle.com/rtatman/english-word-frequency/data
    """
    pairs = []
    with open(frequency_data_file, 'r') as data:
        reader = csv.reader(data, delimiter=',')
        next(reader, None)
        for row in reader:
            word = row[0]
            freq = int(row[1])
            pairs.append((word, freq))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return [p for p in pairs[:len(pairs) if num_of_words > len(pairs) else num_of_words]]


if __name__ == '__main__':
    parser = ArgumentParser(description='Get the most frequent words in English')
    parser.add_argument('--frequency_data_file', type=str)
    parser.add_argument('--num_of_words', type=int)
    args = parser.parse_args()

    frequent_words = get_frequent_words(args.frequency_data_file, args.num_of_words)
    print(frequent_words)