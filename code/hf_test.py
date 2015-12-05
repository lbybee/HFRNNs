from rnn import MetaRNN
from hf import SequenceDataset, hf_optimizer
import numpy as np
import matplotlib.pyplot as plt
import logging
    
    
    
def HFTest(seq, targets, t_seq, t_targets, n_hidden=10, n_updates=250):
    """ Test RNN with hessian free optimization """

    n_in = 2 
    n_out = 2 
    n_classes = 10 

    # SequenceDataset wants a list of sequences
    # this allows them to be different lengths, but here they're not
    seq = [i for i in seq]
    targets = [i for i in targets]

    gradient_dataset = SequenceDataset([seq, targets], batch_size=None,
                                       number_batches=500)
    cg_dataset = SequenceDataset([seq, targets], batch_size=None,
                                 number_batches=100)

    model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    activation='tanh', output_type='softmax',
                    use_symbolic_softmax=True)

    # optimizes negative log likelihood
    # but also reports zero-one error
    opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
                       s=model.rnn.y_pred,
                       costs=[model.rnn.loss(model.y),
                              model.rnn.errors(model.y)], h=model.rnn.h)

    mse_updates = []
    for i in range(n_updates):
        opt.train(gradient_dataset, cg_dataset, num_updates=1)
        mse = 0
        for t in range(len(t_seq)):
            guess = model.predict_proba(t_seq[t])
            if guess != t_target:
                mse += 1
        mse_updates.append(mse)
        print i

    return (mse_updates, model)
