# serverless-inference
Neural network inference on serverless architecture

This repo contains modified implementation of the SM CNN and Kim CNN in the [Castor](https://github.com/castorini/Castor/) repository for serverless deployment on AWS Lambda.

## SM Model

To prepare word vectors to work with the torchtext library, we need to have the word vectors in txt format. Run a command similar to the following for conversion (the script is in the `utils` directory): `python word2vec_bin_to_txt.py ~/castorini/data/word2vec/aquaint+wiki.txt.gz.ndim\=50.bin ~/castorini/data/word2vec/aquaint+wiki.txt.gz.ndim\=50.txt`



# Kim CNN

This implementation is adapted from [Peng's Kim CNN implementation](https://github.com/castorini/Castor/tree/master/kim_cnn)  for serverless-inference experiment.

Peng's Kim CNN implementation has word embedding (i.e. "word to word vectors translation") built-in in the model. But in serverless--inference experiment, we prefer having word embedding stored in DynamoDB for reducing AWS Lambda deployment package size purpose. So the model is modified to take in an "already embedded" word matrix.