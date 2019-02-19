
from __future__ import absolute_import, division

PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'


# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    # Convert sentence to embedding(One dimensional vector)
    return embeddings

if __name__ == "__main__":
	params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
	params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
	                                 'tenacity': 5, 'epoch_size': 4}
	# Set up logger
	se = senteval.engine.SE(params_senteval, batcher, prepare)
	    transfer_tasks = ['MR', 'CR', 'MPQA', 'SUBJ','TREC', 'MRPC',
	                      'SICKEntailment', 'SICKRelatedness']
	    results = se.eval(transfer_tasks)
	    print(results)