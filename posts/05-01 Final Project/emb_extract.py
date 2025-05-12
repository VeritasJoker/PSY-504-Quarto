import argparse
import numpy as np
import pandas as pd
import nltk
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel


def argparse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from reviews")
    parser.add_argument(
        "--data", type=str, default="train", help="Data to process (train/test)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="review",
        help="Mode of processing (title/review)",
    )
    return parser.parse_args()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def main():

    args = argparse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading data
    if args.data == "train":
        df = pd.read_csv("train_s.csv")
    elif args.data == "test":
        df = pd.read_csv("test_s.csv")

    # Loading model
    model = AutoModel.from_pretrained(
        "sentence-transformers/paraphrase-MiniLM-L3-v2", local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/paraphrase-MiniLM-L3-v2", local_files_only=True
    )
    print("Model & Tokenizer loaded")
    sentences = df[args.mode].tolist()

    all_embs = []
    batch_size = 1000
    for i in range(0, len(sentences), batch_size):
        print(
            f"Processing batch {i // batch_size + 1} of {len(sentences) // batch_size + 1}"
        )
        batch_sentences = sentences[i : i + batch_size]
        sentence_inputs = tokenizer(
            batch_sentences, padding=True, truncation=True, return_tensors="pt"
        )

        with torch.no_grad():
            model = model.to(args.device)
            model.eval()
            sentence_inputs = sentence_inputs.to(args.device)
            model_output = model(**sentence_inputs)
            embs = mean_pooling(model_output, sentence_inputs["attention_mask"])
            all_embs.append(embs)
    all_embs = torch.cat(all_embs, dim=0)
    assert all_embs.shape == (len(sentences), model.config.hidden_size)

    np.save(f"{args.data}_{args.mode}_embs.npy", all_embs.cpu().numpy())

    return


if __name__ == "__main__":
    main()
