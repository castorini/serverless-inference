from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Create a text file containing unique words of one or more input text files.')
    parser.add_argument('input', nargs='+')
    parser.add_argument('output')

    args = parser.parse_args()
    print('Input files:', args.input)
    print('Output files:', args.output)

    vocab = set()
    for inp in args.input:
        with open(inp, 'r') as f:
            for line in f:
                words = line.split(' ')
                vocab |= set(words)

    vocab_sorted = sorted(list(vocab))
    with open(args.output, 'w') as f:
        for word in vocab_sorted:
            f.write(word + '\n')
