# serverless-inference
Neural network inference on serverless architecture

This repo contains modified implementation of the SM CNN and Kim CNN in the [Castor](https://github.com/castorini/Castor/) repository for serverless deployment on AWS Lambda.

## SM Model

To prepare word vectors to work with the torchtext library, we need to have the word vectors in txt format. Run a command similar to the following for conversion (the script is in the `utils` directory): `python word2vec_bin_to_txt.py ~/castorini/data/word2vec/aquaint+wiki.txt.gz.ndim\=50.bin ~/castorini/data/word2vec/aquaint+wiki.txt.gz.ndim\=50.txt`
