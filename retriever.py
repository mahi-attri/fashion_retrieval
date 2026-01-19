import json
import argparse
import numpy as np
import faiss
import torch
import clip

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def extract_terms_from_query(q: str, vocab: dict):
    ql = q.lower()
    found = {"color": [], "clothing": [], "environment": [], "style": []}

    def hit(term):
        t = term.lower()
        if t in ql:
            return True
        if " " in t and t in ql:
            return True
        return False

    for term in vocab["colors"]:
        if hit(term):
            found["color"].append(term)
    for term in vocab["clothing"]:
        if hit(term):
            found["clothing"].append(term)
    for term in vocab["environments"]:
        if hit(term):
            found["environment"].append(term)
    for term in vocab["styles"]:
        if hit(term):
            found["style"].append(term)

    return found

class FashionRetriever:
    def __init__(self, index_dir: str, device: str = None, clip_model_name: str = "ViT-B/32"):
        self.index_dir = index_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _ = clip.load(clip_model_name, device=self.device)
        self.model.eval()

        self.index_img = faiss.read_index(f"{index_dir}/index_image.faiss")
        self.index_comb = faiss.read_index(f"{index_dir}/index_combined.faiss")
        self.emb_img = np.load(f"{index_dir}/emb_image.npy").astype(np.float32)
        self.emb_comb = np.load(f"{index_dir}/emb_combined.npy").astype(np.float32)
        self.meta = load_json(f"{index_dir}/metadata.json")
        self.vocab = load_json(f"{index_dir}/vocab.json")

    def encode_text(self, text: str) -> np.ndarray:
        tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
        emb = emb.float().cpu().numpy().astype(np.float32)
        return l2_normalize(emb)[0]

    def build_attr_prompts(self, terms_found: dict):
        templates = self.vocab["templates"]
        prompts = []
        for cat, terms in terms_found.items():
            for t in terms[:3]:
                for tmpl in templates[cat]:
                    prompts.append(tmpl.format(t))
        return prompts

    def rerank(self, q_emb: np.ndarray, candidate_ids, query: str):
        terms_found = extract_terms_from_query(query, self.vocab)
        attr_prompts = self.build_attr_prompts(terms_found)

        attr_embs = None
        if attr_prompts:
            tokens = clip.tokenize(attr_prompts).to(self.device)
            with torch.no_grad():
                te = self.model.encode_text(tokens)
            te = te.float().cpu().numpy().astype(np.float32)
            te = l2_normalize(te)
            attr_embs = te

        scored = []
        for idx in candidate_ids:
            img_v = self.emb_img[idx]
            comb_v = self.emb_comb[idx]

            s_img = float(np.dot(q_emb, img_v))
            s_comb = float(np.dot(q_emb, comb_v))

            s_attr = 0.0
            if attr_embs is not None:
                sims = attr_embs @ img_v
                s_attr = float(np.mean(sims))

            boost = 0.0
            if any(terms_found.values()):
                tops = self.meta[idx].get("top_attributes", {})
                for cat, terms in terms_found.items():
                    top_terms = set([x["term"].lower() for x in tops.get(cat, [])])
                    for t in terms:
                        if t.lower() in top_terms:
                            boost += 0.03

            score = 0.55 * s_comb + 0.35 * s_img + 0.10 * s_attr + boost
            scored.append((score, idx))

        scored.sort(reverse=True, key=lambda x: x[0])
        return scored

    def search(self, query: str, k: int = 5, pre_k: int = 50):
        q_emb = self.encode_text(query)

        D1, I1 = self.index_img.search(q_emb.reshape(1, -1), pre_k)
        D2, I2 = self.index_comb.search(q_emb.reshape(1, -1), pre_k)

        cand = []
        seen = set()
        for i in list(I1[0]) + list(I2[0]):
            if int(i) < 0:
                continue
            if int(i) not in seen:
                seen.add(int(i))
                cand.append(int(i))

        ranked = self.rerank(q_emb, cand, query)[:k]
        results = []
        for score, idx in ranked:
            m = self.meta[idx]
            results.append({"score": score, "path": m["path"], "filename": m["filename"], "id": idx})
        return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", type=str, default="./index_data")
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    r = FashionRetriever(args.index_dir)
    out = r.search(args.query, k=args.k)
    for i, item in enumerate(out, 1):
        print(f"{i}. score={item['score']:.4f} | {item['path']}")

if __name__ == "__main__":
    main()
