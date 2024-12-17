import math
import evaluate
from pycocoevalcap.cider.cider import Cider

def calc_l2(plan, gt):
    l2 = [0.] * 6
    for i, p in enumerate(plan):
        l2[i] += math.sqrt((p[0] - gt[i][0])**2 + (p[1] - gt[i][1])**2)
    return l2

def eval_qa(preds, refs, metric='rouge'):
    # To avoid download config from huggingface, download the metric config on github
    # and load it locally
    # evaluator = evaluate.load('rouge.py')

    # for Cider, simply use
    # evaluator = Cider()
    
    if metric == 'cider':
        predictions, references = {}, {}
        for i, p in enumerate(predictions):
            predictions[i] = p
            references[i] = refs[i]
        evaluator = Cider()
        return {'cider': evaluator.compute_score(references, predictions)[0]}
    
    evaluator = evaluate.load(metric)
    # TODO: lower batch size
    # TODO: support more metrics
    return evaluator.compute(predictions=preds, references=refs)
    