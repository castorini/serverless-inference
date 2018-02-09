# Kim CNN for serverless inferencing on AWS Lambda

This implementation is directly adapted from [Peng's Kim CNN implementation](https://github.com/castorini/Castor/tree/master/kim_cnn) so that the model can be deplyed on AWS Lambda for serverless inferencing.

Peng's Kim CNN implementation has word embedding (i.e. "word to word vectors translation") built-in in the model. But that is not desired in this experiment. This experiment wants to explore the possibility of doing serverless inferencing on a completely serverless architecture by storing the code and model on [AWS Lambda](https://aws.amazon.com/lambda/) and word vectos in [AWS DynamoDB](https://aws.amazon.com/dynamodb/).

In this implementation, the embedding is moved outside the model and wrapped in the `Embedding` class. The model will take in a [PyTorch Variable](http://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_autograd.html) in the forward pass. In addition, the training process is also made to save a CPU-version of model.
