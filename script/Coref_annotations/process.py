#!/usr/bin/env python3
"""
Coréférence sur les entités NER du corpus NerSFcorpus.

Entrée  : data/NerSFcorpus/[auteur]_[titre]_[date]/[auteur]_[titre]_[date].tsv
          (2 colonnes : token, étiquette BIO ; ligne vide = fin de phrase)
          Chemins relatifs au dossier courant ; modifiables avec --input-dir / --output-dir.
Sortie  : data/CorefSFcorpus/[auteur]_[titre]_[date]/[auteur]_[titre]_[date].tsv
          (3 colonnes : token, étiquette BIO, coref)
          coref = {id} sur le premier token de chaque mention, "-" ailleurs.

Principe
--------
1. Les mentions sont les spans BIO du TSV.
2. Le récit est découpé en fenêtres de phrases (budget en tokens) qui se
   chevauchent. Chaque fenêtre passe dans le modèle, qui renvoie des IDs locaux.
3. Les IDs locaux sont réconciliés par union-find sur les mentions :
   - deux mentions de même ID local dans une fenêtre sont fusionnées ;
   - une mention présente dans deux fenêtres (zone de chevauchement) relie
     mécaniquement les clusters des deux fenêtres ;
   - option (activée par défaut) : deux mentions de même forme normalisée
     (casse + ponctuation) et de même type NER sont fusionnées.
4. Les IDs finaux sont renumérotés à partir de 0, dans l'ordre d'apparition.

Le script doit se trouver à côté de coreference.py (preprocess_coreference,
extract_predicted_COREF_ids_from_output).
"""
import argparse
import re
import sys
import time
import unicodedata
from pathlib import Path

PROMPT = '''## Task
Replace each ___ with a numeric coreference ID.

## Rules
- Do NOT modify any other characters in the text.
- The braces { and } must remain unchanged.
- Only replace the substring ___.
- Start numbering from 0 for each document.
- Use the same ID for mentions referring to the same entity.
- Return only the modified text.

## Input
<input_placeholder>

## Output
'''

MAX_SEQ_LEN = 4090
# Tokens ajoutés par mention dans le format "{___ texte}" (estimation prudente)
MENTION_OVERHEAD_TOKENS = 4


# --------------------------------------------------------------------------
# Modèle
# --------------------------------------------------------------------------
class CorefModel:
    def __init__(self, model_name, max_seq_len=MAX_SEQ_LEN):
        from unsloth import FastModel
        from coreference import preprocess_coreference, extract_predicted_COREF_ids_from_output

        self._preprocess = preprocess_coreference
        self._extract = extract_predicted_COREF_ids_from_output

        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_len,
            load_in_4bit=True,
        )
        FastModel.for_inference(self.model)
        self.tokenizer.tokenizer.padding_side = "left"
        self.tok = self.tokenizer.tokenizer
        self.max_seq_len = max_seq_len
        # prompt + balises du chat template + marge
        self.overhead = self.count_tokens(PROMPT.replace("<input_placeholder>", "")) + 32

    def count_tokens(self, text):
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    @staticmethod
    def _max_new_tokens(n_in):
        return max(int(1.25 * n_in), n_in + 20)

    def fits(self, n_in):
        """prompt + entrée + sortie maximale doivent tenir dans max_seq_len."""
        return self.overhead + n_in + self._max_new_tokens(n_in) <= self.max_seq_len

    def prepare(self, document):
        document = self._preprocess(document)
        return document, self.count_tokens(document["input"])

    def predict(self, document):
        n_in = self.count_tokens(document["input"])
        messages = [{"role": "user", "content": PROMPT.replace("<input_placeholder>", document["input"])}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        ).replace("<think>\n\n</think>\n\n", "")
        inputs = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        t0 = time.time()
        out = self.model.generate(
            **inputs,
            max_new_tokens=self._max_new_tokens(n_in),
            do_sample=False,
            use_cache=True,
        )
        dt = max(time.time() - t0, 1e-6)
        n_gen = out.shape[1] - inputs["input_ids"].shape[1]
        print(f"      génération : {n_gen} tokens en {dt:.0f}s ({n_gen / dt:.1f} tok/s, "
              f"max autorisé {self._max_new_tokens(n_in)})", file=sys.stderr, flush=True)
        output = self.tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        ids = self._extract(document["input"], output)
        if len(ids) != len(document["mentions"]):
            raise ValueError(f"{len(ids)} IDs prédits pour {len(document['mentions'])} mentions")
        return ids

    def predict_batch(self, documents):
        """Génère plusieurs fenêtres en même temps (padding à gauche).
        Retourne, pour chaque document, la liste d'IDs ou l'exception rencontrée."""
        prompts, n_ins = [], []
        for d in documents:
            n_ins.append(self.count_tokens(d["input"]))
            messages = [{"role": "user", "content": PROMPT.replace("<input_placeholder>", d["input"])}]
            prompts.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            ).replace("<think>\n\n</think>\n\n", ""))
        inputs = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        max_new = max(self._max_new_tokens(n) for n in n_ins)
        t0 = time.time()
        out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False, use_cache=True)
        dt = max(time.time() - t0, 1e-6)
        gen = out[:, inputs["input_ids"].shape[1]:]
        pad_id = self.tok.pad_token_id
        n_gen = int((gen != pad_id).sum()) if pad_id is not None else gen.numel()
        print(f"      lot de {len(documents)} fenêtres : {n_gen} tokens générés en {dt:.0f}s "
              f"({n_gen / dt:.1f} tok/s au total, max autorisé {max_new})", file=sys.stderr, flush=True)
        outputs = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
        results = []
        for d, text in zip(documents, outputs):
            try:
                ids = self._extract(d["input"], text)
                if len(ids) != len(d["mentions"]):
                    raise ValueError(f"{len(ids)} IDs prédits pour {len(d['mentions'])} mentions")
                results.append(ids)
            except Exception as e:
                results.append(e)
        return results


# --------------------------------------------------------------------------
# Union-find
# --------------------------------------------------------------------------
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# --------------------------------------------------------------------------
# Lecture du TSV, segments, mentions
# --------------------------------------------------------------------------
def read_tsv(path):
    """Retourne une liste de lignes : None (ligne vide) ou (token, étiquette)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip():
                rows.append(None)
            elif "\t" in line:
                token, tag = line.rsplit("\t", 1)
                rows.append((token, tag.strip()))
            else:
                rows.append((line.strip(), "O"))
    return rows


def build_segments(rows, max_words):
    """Phrases (blocs de lignes non vides), découpées si plus longues que
    max_words, sans jamais couper à l'intérieur d'une entité (avant un I-)."""
    segments, current = [], []
    for i, row in enumerate(rows):
        if row is None:
            if current:
                segments.append(current)
                current = []
            continue
        if len(current) >= max_words and not row[1].startswith("I-"):
            segments.append(current)
            current = []
        current.append(i)
    if current:
        segments.append(current)
    return segments


def extract_mentions(rows, segments):
    mentions = []
    for seg_id, seg in enumerate(segments):
        cur = None
        for pos, row_idx in enumerate(seg):
            tag = rows[row_idx][1]
            if tag.startswith("B-") or tag.startswith("I-"):
                etype = tag[2:]
                if tag.startswith("I-") and cur is not None and cur["type"] == etype:
                    cur["end"] = pos
                else:
                    cur = {"seg": seg_id, "start": pos, "end": pos, "type": etype}
                    mentions.append(cur)
            else:
                cur = None
    for i, m in enumerate(mentions):
        seg = segments[m["seg"]]
        m["idx"] = i
        m["row"] = seg[m["start"]]  # ligne du premier token (où l'on écrira {id})
        m["text"] = " ".join(rows[r][0] for r in seg[m["start"]:m["end"] + 1])
    return mentions


def segment_texts(rows, segments, mentions):
    """Texte de chaque segment (tokens séparés par un espace) et offsets
    relatifs des mentions dans ce texte."""
    by_seg = [[] for _ in segments]
    for m in mentions:
        by_seg[m["seg"]].append(m)

    texts = []
    for seg_id, seg in enumerate(segments):
        tokens = [rows[r][0] for r in seg]
        starts, pos = [], 0
        for t in tokens:
            starts.append(pos)
            pos += len(t) + 1
        texts.append(" ".join(tokens))
        for m in by_seg[seg_id]:
            m["rel_onset"] = starts[m["start"]]
            m["rel_offset"] = starts[m["end"]] + len(tokens[m["end"]])
    return texts, by_seg


# --------------------------------------------------------------------------
# Fenêtrage
# --------------------------------------------------------------------------
def make_windows(costs, budget, overlap_tokens):
    """Fenêtres [a, b) de segments consécutifs dont le coût cumulé <= budget.
    La fenêtre suivante reprend les derniers segments (jusqu'à overlap_tokens)."""
    windows, n, a = [], len(costs), 0
    while a < n:
        b, total = a, 0
        while b < n and (b == a or total + costs[b] <= budget):
            total += costs[b]
            b += 1
        windows.append((a, b))
        if b >= n:
            break
        s, ov = b, 0
        while s > a + 1 and ov + costs[s - 1] <= overlap_tokens:
            s -= 1
            ov += costs[s]
        a = s
    return windows


def build_document(seg_texts, seg_mentions, a, b):
    parts, mentions, order, shift = [], [], [], 0
    for s in range(a, b):
        for m in seg_mentions[s]:
            mentions.append({
                "COREF": "0",
                "onset": shift + m["rel_onset"],
                "offset": shift + m["rel_offset"],
                "text": m["text"],
            })
            order.append(m["idx"])
        parts.append(seg_texts[s])
        shift += len(seg_texts[s]) + 1
    return {"text": " ".join(parts), "mentions": mentions}, order


def collect_windows(resolver, seg_texts, seg_mentions, costs, a, b, budget, overlap, stats, items):
    """Construit les fenêtres sur les segments [a, b) et les ajoute à `items`
    sous forme (document, ordre des mentions). Si une fenêtre dépasse la limite
    réelle de tokens (mesurée sur l'entrée préprocessée), elle est redécoupée plus petit."""
    for wa, wb in make_windows(costs[a:b], budget, int(budget * overlap)):
        wa, wb = wa + a, wb + a
        if sum(len(seg_mentions[s]) for s in range(wa, wb)) == 0:
            continue
        doc, order = build_document(seg_texts, seg_mentions, wa, wb)
        doc, n_in = resolver.prepare(doc)

        if not resolver.fits(n_in):
            if wb - wa > 1 and budget > 100:
                collect_windows(resolver, seg_texts, seg_mentions, costs, wa, wb,
                                int(budget * 0.7), overlap, stats, items)
            else:
                stats["failed"] += 1
                print(f"    ! fenêtre trop longue ({n_in} tokens), ignorée", file=sys.stderr)
            continue

        print(f"    fenêtre phrases {wa}-{wb} : {n_in} tokens d'entrée, {len(order)} mentions",
              file=sys.stderr, flush=True)
        items.append((doc, order))


def _safe(fn, *args):
    try:
        return fn(*args)
    except Exception as e:
        return e


def _free_gpu():
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def predict_many(resolver, docs, batch_size):
    """Une liste (IDs ou exception) par document. Un lot qui échoue (mémoire...)
    est rejoué fenêtre par fenêtre ; une fenêtre mal formée dans un lot est rejouée seule."""
    results = []
    for i in range(0, len(docs), batch_size):
        chunk = docs[i:i + batch_size]
        res = None
        if batch_size > 1 and len(chunk) > 1 and hasattr(resolver, "predict_batch"):
            try:
                res = resolver.predict_batch(chunk)
            except Exception as e:
                print(f"    ! échec du lot ({type(e).__name__}: {e}), repli fenêtre par fenêtre",
                      file=sys.stderr, flush=True)
                _free_gpu()
        if res is None:
            res = [_safe(resolver.predict, d) for d in chunk]
        else:
            res = [_safe(resolver.predict, d) if isinstance(r, Exception) else r
                   for r, d in zip(res, chunk)]
        if any(isinstance(r, Exception) for r in res):
            _free_gpu()
        results.extend(res)
    return results


def run_windows(resolver, seg_texts, seg_mentions, costs, a, b, budget, overlap, uf, stats,
                batch_size=1):
    """Fenêtres sur les segments [a, b) -> prédictions (par lots) -> fusion des IDs locaux."""
    items = []
    collect_windows(resolver, seg_texts, seg_mentions, costs, a, b, budget, overlap, stats, items)
    results = predict_many(resolver, [doc for doc, _ in items], batch_size)

    for (doc, order), ids in zip(items, results):
        if isinstance(ids, Exception):
            stats["failed"] += 1
            print(f"    ! échec sur une fenêtre ({type(ids).__name__}: {ids})", file=sys.stderr)
            continue
        stats["windows"] += 1
        first = {}
        for midx, lid in zip(order, ids):
            if lid in first:
                uf.union(first[lid], midx)
            else:
                first[lid] = midx


# --------------------------------------------------------------------------
# Fusion par nom
# --------------------------------------------------------------------------
def normalize(text):
    """Casse et ponctuation neutralisées (les accents sont conservés)."""
    t = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\W_]+", "", t)


def merge_same_names(mentions, uf):
    seen = {}
    for m in mentions:
        key = (normalize(m["text"]), m["type"])
        if not key[0]:
            continue
        if key in seen:
            uf.union(seen[key], m["idx"])
        else:
            seen[key] = m["idx"]


# --------------------------------------------------------------------------
# Un récit
# --------------------------------------------------------------------------
def process_story(in_path, out_path, resolver, window_tokens=1400, overlap=0.25,
                  max_segment_words=120, merge_by_name=True, batch_size=1):
    rows = read_tsv(in_path)
    segments = build_segments(rows, max_segment_words)
    mentions = extract_mentions(rows, segments)
    seg_texts, seg_mentions = segment_texts(rows, segments, mentions)

    uf = UnionFind(len(mentions))
    stats = {"mentions": len(mentions), "windows": 0, "failed": 0}

    if mentions:
        costs = [resolver.count_tokens(t) + MENTION_OVERHEAD_TOKENS * len(ms)
                 for t, ms in zip(seg_texts, seg_mentions)]
        n_win = len(make_windows(costs, window_tokens, int(window_tokens * overlap)))
        print(f"    {len(segments)} phrases, {len(mentions)} mentions, ~{sum(costs)} tokens "
              f"-> ~{n_win} fenêtres", file=sys.stderr, flush=True)
        run_windows(resolver, seg_texts, seg_mentions, costs, 0, len(segments),
                    window_tokens, overlap, uf, stats, batch_size=batch_size)
        if merge_by_name:
            merge_same_names(mentions, uf)

    # IDs finaux : 0, 1, 2... dans l'ordre d'apparition
    root_to_id = {}
    start_row_to_id = {}
    for m in mentions:
        root = uf.find(m["idx"])
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        start_row_to_id[m["row"]] = root_to_id[root]
    stats["clusters"] = len(root_to_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tsv.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            if row is None:
                f.write("\n")
            else:
                coref = f"{{{start_row_to_id[i]}}}" if i in start_row_to_id else "-"
                f.write(f"{row[0]}\t{row[1]}\t{coref}\n")
    tmp.replace(out_path)
    return stats


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="data/NerSFcorpus",
                    help="dossier des récits (défaut : data/NerSFcorpus, relatif au dossier courant)")
    ap.add_argument("--output-dir", default="data/CorefSFcorpus")
    ap.add_argument("--model", default="lattice-nlp/Qwen3.5-4B_inline_coreference_resolution")
    ap.add_argument("--window-tokens", type=int, default=1400,
                    help="budget d'entrée par fenêtre, en tokens (défaut 1400)")
    ap.add_argument("--overlap", type=float, default=0.25,
                    help="part du budget reprise d'une fenêtre à la suivante (défaut 0.25)")
    ap.add_argument("--max-segment-words", type=int, default=120,
                    help="longueur max d'une phrase avant découpage forcé (défaut 120 tokens TSV)")
    ap.add_argument("--no-merge-by-name", action="store_true",
                    help="désactive la fusion des mentions de même forme normalisée et de même type")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="nombre de fenêtres d'un même récit générées en même temps (défaut 1 ; "
                         "essayer 3 ou 4 et vérifier que le résultat reste cohérent)")
    ap.add_argument("--overwrite", action="store_true", help="recalcule les récits déjà présents en sortie")
    ap.add_argument("--limit", type=int, default=None, help="ne traiter que N récits (test)")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    files = sorted(in_dir.glob("*/*.tsv"))
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit(f"Aucun TSV trouvé dans {in_dir.resolve()}/*/*.tsv "
                 f"(dossier courant : {Path.cwd()}). Utilise --input-dir pour indiquer le bon chemin.")

    resolver = CorefModel(args.model)
    print(f"Limite d'entrée réelle : fenêtres vérifiées contre {MAX_SEQ_LEN} tokens "
          f"(prompt ≈ {resolver.overhead} tokens)")

    for k, tsv in enumerate(files, 1):
        out = out_dir / tsv.relative_to(in_dir)
        if out.exists() and not args.overwrite:
            print(f"[{k}/{len(files)}] {tsv.stem} : déjà fait, ignoré")
            continue
        try:
            st = process_story(tsv, out, resolver,
                               window_tokens=args.window_tokens,
                               overlap=args.overlap,
                               max_segment_words=args.max_segment_words,
                               merge_by_name=not args.no_merge_by_name,
                               batch_size=args.batch_size)
        except Exception as e:
            print(f"[{k}/{len(files)}] {tsv.stem} : ERREUR {type(e).__name__}: {e}", file=sys.stderr)
            continue
        print(f"[{k}/{len(files)}] {tsv.stem} : {st['mentions']} mentions, "
              f"{st['windows']} fenêtres ({st['failed']} échecs), {st['clusters']} clusters")


if __name__ == "__main__":
    main()