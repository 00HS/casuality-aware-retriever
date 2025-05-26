import torch

def negative_log_loss(causal_vectors, semantic_vectors, device):

        scores = torch.matmul(causal_vectors, torch.transpose(semantic_vectors, 0, 1))

        if len(causal_vectors.size()) > 1:
            q_num = causal_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = torch.nn.functional.log_softmax(scores, dim=1)
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=device)

        loss = torch.nn.functional.nll_loss(
            softmax_scores,
            labels,
            reduction="mean",
        )

        return loss