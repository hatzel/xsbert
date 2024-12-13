from collections import defaultdict
import gc
import os
import torch
import json
import sentence_transformers
from sentence_transformers.models import Pooling
from xsbert import models, utils
from typer import Typer
from pathlib import Path
from typing import Optional
from datasets import load_dataset
import spacy
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots

app = Typer()

prefix = "Retrieve stories with a similar narrative to the given story: "

def get_model(model_name: str):
    transformer = models.ReferenceTransformer(model_name)
    pooling = Pooling(transformer.get_word_embedding_dimension())
    if "all-mpnet" in model_name:
        model = models.XSMPNet(modules=[transformer, pooling])
    if "roberta" in model_name:
        model = models.XSRoberta(modules=[transformer, pooling])
    elif "mistral" in model_name or "nar-emb" in model_name:
        pooling = Pooling(transformer.get_word_embedding_dimension(), pooling_mode="lasttoken")
        model = models.XMistral(modules=[transformer, pooling])
        transformer.tokenizer.pad_token = transformer.tokenizer.eos_token
        transformer.tokenizer.padding_side = "left"
    model.to("cuda:0")
    return model


@app.command()
def compute(text_file_a: str, text_file_b: str, output_path: str, model_name: str = "intfloat/e5-mistral-7b-instruct", layer: int = 31):
    # model_name = "sentence-transformers/all-distilroberta-v1"
    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # model_name = "intfloat/e5-mistral-7b-instruct"


    model = get_model(model_name)

    text_a = prefix + "".join(open(text_file_a, "r").readlines()).strip()
    text_b = prefix + "".join(open(text_file_b, "r").readlines()).strip()
    prefix_tokens = model.tokenizer(prefix).tokens()

    encoded = model.encode([text_a, text_b])
    sims = sentence_transformers.util.cos_sim(encoded, encoded)
    sims = sims.cpu()


    model.reset_attribution()
    # There are 32 layers in e5
    # for x in range(32):
    model.init_attribution_to_layer(idx=layer, N_steps=50)

    A, tokens_a, tokens_b = model.explain_similarity(
        text_a, 
        text_b, 
        move_to_cpu=True,
        sim_measure='cos',
    )
    # ../nar-embv02 torch.Size([12, 13])
    # intfloat/e5-mistral-7b-instruct torch.Size([13, 14])
    print(model_name, A.shape)
    print("length a:", len(tokens_a), "length b:", len(tokens_b))
    json.dump({
        "matrix": [[e.item() for e in row] for row in A],
        "layer": layer,
        "model": model_name,
        "prefix_tokens": prefix_tokens,
        "tokens_a": tokens_a,
        "tokens_b": tokens_b,
        "similarity": sims[0,1].item(),
    }, open(output_path, "w"))


@app.command()
def visualize(input_path: str, output_path: str, compare_to: Optional[str] = None, zero_corner: bool = False, zero_edge: bool = False, skip_first: int = 13, remove_edge: bool = False):
    data = json.load(open(Path(input_path)))
    A = torch.tensor(data["matrix"])[skip_first:,skip_first:]
    B = None
    if compare_to is not None:
        data_b = json.load(open(Path(compare_to)))
        B = torch.tensor(data_b["matrix"])[skip_first:,skip_first:]
        assert A.shape == B.shape
        overall = A - B
    else:
        overall = A
    tokens_a = data["tokens_a"][skip_first:]
    tokens_b = data["tokens_b"][skip_first:]
    if zero_corner:
        A[-1,-1] = 0
    if zero_edge:
        A[:,-1] = 0
        A[-1,:] = 0
    if remove_edge:
        overall = overall[:-1,:-1]
        A = A[:-1,:-1]
        tokens_a = tokens_a[:-1]
        tokens_b = tokens_b[:-1]
        if B is not None:
            B = B[:-1,:-1]

    # plt.style.use('science')
    f = utils.plot_attributions(
        overall, 
        tokens_a, 
        tokens_b, 
        size=(5, 5),
        # range=.3,
        show_colorbar=True, 
        #shrink_cbar=.5
    )
    f.savefig(output_path)



@app.command()
def tags(model_name: str = "intfloat/e5-mistral-7b-instruct", layer: int = 31):
    model = get_model(model_name)
    nlp = spacy.load("en_core_web_lg")

    similarities = defaultdict(list)
    similarities_pos = defaultdict(list)
    prefix_doc = nlp(prefix)
    # prefix = ""
    dataset = load_dataset("stsb_multi_mt", name="en", split="test").shuffle(seed=42)
    print("Num items", len(dataset))
    max_len = 25
    for i, row in tqdm(enumerate(dataset)):
        if i > 50:
            break
        text_a, text_b = row["sentence1"], row["sentence2"]
        text_a = prefix + text_a
        text_b = prefix + text_b
        model.reset_attribution()
        model.init_attribution_to_layer(idx=layer, N_steps=50)
        doc_a = nlp(text_a)
        doc_b = nlp(text_b)
        print(len(doc_a))
        if len(doc_a) + len(doc_b) > max_len * 2:
            continue

        A, tokens_a, tokens_b = model.explain_similarity(
            text_a, 
            text_b, 
            move_to_cpu=True,
            sim_measure='cos',
            verbose=False,
        )
        skipped = 0
        A.detach()

        # a_tokenized = model.tokenizer(text_a)
        # for token in doc_a:
        #     tokens = [a_tokenized.char_to_token(x) for x in range(token.idx, token.idx + len(token.text)) if a_tokenized.char_to_token(x) is not None]
        #     similarities[token.ent_type_].extend([e.item() for e in A[tokens,:].sum(1)])
        # b_tokenized = model.tokenizer(text_a)
        for doc, tokenized, is_first in zip([doc_a, doc_b], [model.tokenizer(t) for t in [text_a, text_b]], [True, False]):
            for i, token in enumerate(doc):
                if i <= len(prefix_doc):
                    pass
                tokens = [tokenized.char_to_token(x) for x in range(token.idx, token.idx + len(token.text)) if tokenized.char_to_token(x) is not None]
                tokens = list(set(tokens))
                if is_first:
                    new_scores = [e.item() for e in A[tokens,:].sum(1)]
                else:
                    new_scores = [e.item() for e in A[:,tokens].sum(1)]
                similarities[token.ent_type_].extend(new_scores)
                similarities_pos[token.pos_].extend(new_scores)
        del A, tokens_a, tokens_b
        gc.collect()

    out_file = open(model_name.replace("/", "--").replace(".", "") + str(layer) + ".txt", "w")
    print("Skipped:", skipped, file=out_file)
    print("ENT", file=out_file)
    for k, v in similarities.items():
        print(k, f"{sum(v) / len(v):.4f}", file=out_file)

    print("POS", file=out_file)
    for k, v in similarities_pos.items():
        print(k, f"{sum(v) / len(v):.4f}", file=out_file)
    # print(a_tokenized)
    # breakpoint()
    # print(a_tokenized)

if __name__ == "__main__":
    app()
