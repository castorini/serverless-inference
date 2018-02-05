import torch
import numpy
import pickle
from torch.autograd import Variable


model = torch.load('saves/SST-1/non-static_best_model.pt')
model.eval()
dataset_file = 'data/word2vec.sst-1.pt' # file to load from
wordindex_file = 'data/wordindex.pkl' # file to save to
indexvec_file = 'data/indexvec.npy' # file to save to


with open(wordindex_file, 'rb') as f:
    wordindex = pickle.load(f)
indexvec = numpy.load(indexvec_file)


sentence = 'this is very very good movie like it very much'
words = sentence.split(' ')
matrix = None
for i in range(50):
	if i < len(words):
		if words[i] in wordindex:
			vec = indexvec[wordindex[words[i]]]
		else:
			print(words[i])
			vec = numpy.random.rand(300)
	else:
		vec = numpy.zeros(300)
	vec.resize(1, 1, vec.shape[0])
	if matrix is None:
		matrix = vec
	else:
		matrix = numpy.append(matrix, vec, axis=1)

matrix.resize(1, matrix.shape[0], matrix.shape[1], matrix.shape[2])
matrix = torch.from_numpy(matrix.astype(numpy.float32))
matrix = Variable(matrix)
print(matrix.shape)
print(model(matrix))
print(torch.max(model(matrix), 1)[1])


print(indexvec[wordindex['good']])