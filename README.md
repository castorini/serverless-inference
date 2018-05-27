# serverless-inference

Neural network inference on serverless architecture

This repo contains modified implementations of [SM-CNN](https://github.com/castorini/Castor/tree/master/sm_cnn) and [Kim CNN](https://github.com/castorini/Castor/tree/master/kim_cnn) in the [Castor](https://github.com/castorini/Castor/) repository for serverless deployment on AWS Lambda.

This project uses Amazon DynamoDB for storing word vectors and AWS Lambda for inference.

## Instructions for Deploying Model on AWS

### Upload word vectors to DynamoDB

The original SM-CNN paper and Kim CNN paper use word2vec and GloVe word vectors respectively. The `utils/upload_word_vectors_to_dynamo.py` script can be used to upload these word vectors to DynamoDB.

Uploading the word vectors for the entire word2vec or GloVe vocabulary can be too much. Hence, we can use the `utils/get_most_frequent_words.py` script to generate a filtered list of words that appear the most often from a word frequency dataset (see script for more info). Alternatively, we can also generate a filtered list of words from a set of unique words that appear in some dataset on [Castor-data](https://git.uwaterloo.ca/jimmylin/Castor-data) using `utils/generate_vocab.py` by passing in the paths to {a,b}.toks for some dataset.

For example:

```bash
$ python generate_vocab.py /Castor-data/TrecQA/train/a.toks /Castor-data/TrecQA/train/b.toks TrecQA-train-vocab.txt
```

The `utils/upload_word_vectors_to_dynamo.py` script takes in the word vectors file in txt format. Hence if the word vectors are in binary format we need to convert it, with the help of the `utils/word2vec_bin_to_txt.py` script.

For example:

```bash
$ python word2vec_bin_to_txt.py /Castor-data/embeddings/word2vec/aquaint+wiki.txt.gz.ndim\=50.bin /Castor-data/embeddings/word2vec/aquaint+wiki.txt.gz.ndim\=50.txt
```

After we create a table on DynamoDB, we can then upload word vectors to the table. The primary key of the table should be named `word`.

For example:

```bash
$ python upload_word_vectors_to_dynamo.py /Castor-data/embeddings/word2vec/aquaint+wiki.txt.gz.ndim\=50.txt word2vec --wordlist TrecQA-train-vocab.txt
```

### Building Deployment Package

Create a EC2 instance with the Amazon Linux AMI to build the Lambda deployment package. We used a t2.medium instance with 4 GB of memory.

Once you ssh into the EC2 instance, use `utils/create_deployment_package.sh` to build the deployment package.

Then copy the deployment package to your machine from EC2.

### Train Model and Bundle with Deployment Package

Train the SM-CNN and Kim CNN models following instructions in [Castor](https://github.com/castorini/Castor/).

Extract the deployment package into some directory. Copy the serialized model (`{sm_cnn,kim_cnn}.model`) and code (model source file and handler) into the directory.

For SM-CNN, the new files you copy to the unzipped deployment package should follow this structure:

```
.
├── sm_cnn
|      ├── __init__.py
|      └── model.py
├── sm_handler.py
└── sm_cnn.model
```

For Kim CNN, the new files you copy to the unzipped deployment package should follow this structure:

```
.
├── kim_cnn
|      ├── __init__.py
|      └── model.py
├── kim_handler.py
└── kim_cnn.model
```

Finally, zip up the new deployment package with the model and code.

### Create Lambda Function

Upload the deployment package to some bucket on S3.

Create a new Lambda function and specify the S3 link.

Here is an example SM-CNN test event:

```json
{
  "body": "{\"sent1\": \"how are glacier caves formed\", \"sent2\": \"the ice facade is approximately 60 m high\"}"
}
```

Here is an example Kim CNN test event:

```json
{
  "body": "{\"input\": \"this is a very good movie\"}"
}
```

You can take the Lambda function invokable using a REST API by adding a AWS API Gateway trigger.
