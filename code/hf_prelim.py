from rnn import MetaRNN
from hf import SequenceDataset, hf_optimizer
import numpy as np
import matplotlib.pyplot as plt
import logging
    
    
    
def HFTest(n_updates=100):
    """ Test RNN with hessian free optimization """

    n_hidden = 10
    n_in = 5
    n_steps = 10
    n_seq = 1000
    n_test = 100
    n_classes = 3
    n_out = n_classes  # restricted to single softmax per time step

    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_seq, n_steps, n_in)
    targets = np.zeros((n_seq, n_steps), dtype='int32')

    thresh = 0.5
    # if lag 1 (dim 3) is greater than lag 2 (dim 0) + thresh
    # class 1
    # if lag 1 (dim 3) is less than lag 2 (dim 0) - thresh
    # class 2
    # if lag 2(dim0) - thresh <= lag 1 (dim 3) <= lag2(dim0) + thresh
    # class 0
    targets[:, 2:][seq[:, 1:-1, 3] > seq[:, :-2, 0] + thresh] = 1
    targets[:, 2:][seq[:, 1:-1, 3] < seq[:, :-2, 0] - thresh] = 2
    #targets[:, 2:, 0] = np.cast[np.int](seq[:, 1:-1, 3] > seq[:, :-2, 0])

    # SequenceDataset wants a list of sequences
    # this allows them to be different lengths, but here they're not
    seq = [i for i in seq]
    targets = [i for i in targets]

    t_seq = seq[-n_test:]
    t_targets = targets[-n_test:]
    seq = seq[:-n_test]
    targets = targets[:-n_test]

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
            guess = np.array([list(r).index(max(r)) for r in guess])
            mse += sum(guess != np.array(t_targets[t]))
        mse_updates.append(mse)
        print i

    seqs = xrange(10)

    plt.close('all')
    for seq_num in seqs:
        fig = plt.figure()
        ax1 = plt.subplot(211)
        plt.plot(seq[seq_num])
        ax1.set_title('input')

        ax2 = plt.subplot(212)
        # blue line will represent true classes
        true_targets = plt.step(xrange(n_steps), targets[seq_num], marker='o')

        # show probabilities (in b/w) output by model
        guess = model.predict_proba(seq[seq_num])
        guessed_probs = plt.imshow(guess.T, interpolation='nearest',
                                   cmap='gray')
        ax2.set_title('blue: true class, grayscale: probs assigned by model')
    fig.savefig("results.jpg")

    plt.close("all")
    fig = plt.figure()
    plt.plot(mse_updates)
    plt.ylabel("Error Count")
    plt.xlabel("Updates")
    fig.savefig("errors.jpg")

    return (mse_updates, model)

