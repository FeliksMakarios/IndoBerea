"""Show that the v1 training setup produces no learning signal.

The original notebook built `InputExample(texts=[verse])` without a label, so every
example has label 0, and trained with `BatchHardTripletLoss`. With a single label in
the batch there are no negatives: the loss is constant (equal to the margin, 5.0) and
the gradient is zero. A tiny randomly initialized BERT is enough to show this, so the
script runs on CPU in seconds and needs no download.

    python scripts/check_batch_hard_loss.py
"""
import os
import tempfile

import torch
from sentence_transformers import InputExample, SentenceTransformer, losses, models
from transformers import BertConfig, BertModel, BertTokenizerFast


def tiny_model():
    d = tempfile.mkdtemp()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + (
        "jangan berzinah membunuh mencuri kasih allah dosa manusia akulah roti hidup tuhan yesus".split()
    )
    with open(os.path.join(d, "vocab.txt"), "w") as f:
        f.write("\n".join(vocab))
    BertTokenizerFast(os.path.join(d, "vocab.txt")).save_pretrained(d)
    config = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=2, intermediate_size=64)
    BertModel(config).save_pretrained(d)
    transformer = models.Transformer(d)
    return SentenceTransformer(modules=[transformer, models.Pooling(32, "mean")])


def loss_and_grad(model, texts, labels):
    model.zero_grad()
    loss_fn = losses.BatchHardTripletLoss(model=model)
    loss = loss_fn([model.tokenize(texts)], torch.tensor(labels))
    loss.backward()
    grad = sum(p.grad.norm() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    return loss.item(), float(grad)


def main():
    torch.manual_seed(0)
    model = tiny_model()
    texts = ["jangan berzinah", "kasih allah", "dosa manusia", "akulah roti hidup", "tuhan yesus", "jangan membunuh"]

    # Exactly as in the original notebook: one text per example, no label.
    examples = [InputExample(texts=[t]) for t in texts]
    labels = [e.label for e in examples]
    print("labels used by the notebook:", labels)
    loss, grad = loss_and_grad(model, texts, labels)
    print(f"v1 setup        -> loss {loss:.4f}, gradient norm {grad:.4f}")

    # For comparison: with more than one label the loss does produce a gradient.
    loss, grad = loss_and_grad(model, texts, [0, 1, 2, 3, 1, 0])
    print(f"varied labels   -> loss {loss:.4f}, gradient norm {grad:.4f}")


if __name__ == "__main__":
    main()
