import json
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import clip
import faiss

COLORS = [
    "black","white","red","blue","green","yellow","orange","pink","purple","brown","beige","gray","grey",
    "navy","maroon","teal","turquoise","cream","gold","silver"
]

CLOTHING = [
    "t-shirt","shirt","button-down shirt","blouse","hoodie","sweater","jacket","coat","raincoat","blazer",
    "suit","dress","skirt","jeans","pants","trousers","shorts","tie","sneakers","shoes","boots","heels"
]

ENVIRONMENTS = [
    "modern office","office interior","office","urban street","city street","street","park","park bench",
    "home","living room","indoors","outdoors","cafe","mall"
]

STYLES = [
    "formal","business","professional","business casual","smart casual","casual","streetwear","weekend",
    "athleisure"
]

TEMPLATES = {
    "color": [
        "a photo of a person wearing a {} outfit",
        "a person in {} clothing",
        "a {} colored garment"
    ],
    "clothing": [
        "a photo of a person wearing a {}",
        "a person in a {}",
        "a {} garment"
    ],
    "environment": [
        "a photo taken inside a {}",
        "a person in a {}",
        "a scene in a {}"
    ],
    "style": [
        "a photo of {} style outfit",
        "a person dressed in {} fashion",
        "{} attire"
    ]
}

def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)

class FashionIndexer:
    def __init__(self, device: str = None, clip_model_name: str = "ViT-B/32"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print("Loading CLIP model...")
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device)
        self.model.eval()
        self.text_bank = self._build_text_bank()

    def _encode_text(self, texts):
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
        emb = emb.float().cpu().numpy().astype(np.float32)
        return l2_normalize(emb)

    def _encode_image(self, image: Image.Image):
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(x)
        emb = emb.float().cpu().numpy().astype(np.float32)
        return l2_normalize(emb)[0]

    def _build_text_bank(self):
        bank = {}
        for cat, vocab in [
            ("color", COLORS),
            ("clothing", CLOTHING),
            ("environment", ENVIRONMENTS),
            ("style", STYLES),
        ]:
            phrases = []
            phrase_meta = []
            for term in vocab:
                for tmpl in TEMPLATES[cat]:
                    phrases.append(tmpl.format(term))
                    phrase_meta.append({"category": cat, "term": term, "phrase": tmpl.format(term)})
            text_emb = self._encode_text(phrases)
            bank[cat] = {"phrases": phrases, "meta": phrase_meta, "emb": text_emb}
        return bank

    def _top_terms_for_category(self, image_emb: np.ndarray, category: str, topk_terms: int = 3):
        tb = self.text_bank[category]
        sims = (tb["emb"] @ image_emb).astype(np.float32)
        best_for_term = {}
        for i, m in enumerate(tb["meta"]):
            term = m["term"]
            s = float(sims[i])
            if term not in best_for_term or s > best_for_term[term]["score"]:
                best_for_term[term] = {"score": s, "phrase": m["phrase"], "term": term}
        ranked = sorted(best_for_term.values(), key=lambda x: x["score"], reverse=True)[:topk_terms]
        return ranked

    def _build_attribute_fusion_vector(self, top_attrs):
        phrases = []
        for cat in ["color","clothing","environment","style"]:
            for item in top_attrs.get(cat, []):
                phrases.append(item["phrase"])
        if not phrases:
            return None
        text_embs = self._encode_text(phrases)
        fused = l2_normalize(text_embs.mean(axis=0, keepdims=False).astype(np.float32))
        return fused

    def process_dataset(self, image_dir: str, output_dir: str = "./index_data", max_images: int = None):
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        exts = {".jpg",".jpeg",".png",".bmp",".webp"}
        paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in exts]
        if max_images is not None:
            paths = paths[:max_images]

        print(f"Found {len(paths)} images to process")

        img_vecs = []
        combined_vecs = []
        metadata = []

        for p in tqdm(paths, desc="Indexing"):
            try:
                im = Image.open(p).convert("RGB")
                img_emb = self._encode_image(im)

                top_attrs = {
                    "color": self._top_terms_for_category(img_emb, "color", topk_terms=3),
                    "clothing": self._top_terms_for_category(img_emb, "clothing", topk_terms=3),
                    "environment": self._top_terms_for_category(img_emb, "environment", topk_terms=3),
                    "style": self._top_terms_for_category(img_emb, "style", topk_terms=3),
                }

                attr_fused = self._build_attribute_fusion_vector(top_attrs)
                if attr_fused is None:
                    combined = img_emb
                else:
                    combined = l2_normalize((0.75 * img_emb + 0.25 * attr_fused).astype(np.float32))

                img_vecs.append(img_emb)
                combined_vecs.append(combined)

                metadata.append({
                    "id": len(metadata),
                    "path": str(p),
                    "filename": p.name,
                    "top_attributes": {
                        k: [{"term": a["term"], "score": a["score"]} for a in v] for k, v in top_attrs.items()
                    }
                })

            except Exception as e:
                print(f"Failed on {p}: {e}")

        img_vecs = np.asarray(img_vecs, dtype=np.float32)
        combined_vecs = np.asarray(combined_vecs, dtype=np.float32)

        if len(metadata) == 0:
            raise RuntimeError("No images indexed. Check IMAGE_DIR and file extensions.")

        d = img_vecs.shape[1]

        index_img = faiss.IndexFlatIP(d)
        index_img.add(img_vecs)

        index_combined = faiss.IndexFlatIP(d)
        index_combined.add(combined_vecs)

        faiss.write_index(index_img, str(output_dir / "index_image.faiss"))
        faiss.write_index(index_combined, str(output_dir / "index_combined.faiss"))

        np.save(output_dir / "emb_image.npy", img_vecs)
        np.save(output_dir / "emb_combined.npy", combined_vecs)

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        bank_save = {
            "colors": COLORS,
            "clothing": CLOTHING,
            "environments": ENVIRONMENTS,
            "styles": STYLES,
            "templates": TEMPLATES
        }
        with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(bank_save, f, indent=2)

        print("Indexing complete")
        print(f"Images indexed: {len(metadata)}")
        print(f"Saved to: {output_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./index_data")
    ap.add_argument("--max_images", type=int, default=None)
    args = ap.parse_args()

    indexer = FashionIndexer()
    indexer.process_dataset(args.image_dir, args.output_dir, args.max_images)

if __name__ == "__main__":
    main()
