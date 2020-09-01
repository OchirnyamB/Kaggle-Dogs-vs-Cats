import numpy as np 

def rank_accuracy(preds, labels):
    rank1 = 0
    rank5 = 0

    # loop over the predictions and ground-truth labels
    for(p, gt) in zip(preds, labels):
        # sort probabilities by their index in descending order
        p = np.argsort(p)[::-1]

        # check to see if the ground-truth label is in the top-5
        if gt in p[:5]:
            rank5 += 1
        # check to see if the ground-truth label is the #1 prediction
        if gt == p[0]:
            rank1 += 1 
    
    # compute the final rank-1 and rank-5 accuracies
    rank1 /= float(len(preds))
    rank5 /= float(len(preds))

    return (rank1, rank5)