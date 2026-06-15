import torch
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:",DEVICE)

def compute_metrics(texts, topics):
    topics = {"topics": topics}
    cvmetric = Coherence(texts = texts, topk=10, measure='c_v')
    c_v = cvmetric.score(topics)
    topic_diversity = TopicDiversity(topk=25)
    td = topic_diversity.score(topics)
    cnmetric = Coherence(texts = texts, topk=10, measure='c_npmi')
    c_npmi = cnmetric.score(topics)
    return [c_v,c_npmi,td]
