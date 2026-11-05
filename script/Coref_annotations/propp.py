from unsloth import FastModel

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

model, tokenizer = FastModel.from_pretrained(
    model_name="lattice-nlp/Qwen3.5-4B_inline_coreference_resolution",
    max_seq_length=4090,
    load_in_4bit=True,
)
FastModel.for_inference(model)
tokenizer.tokenizer.padding_side = "left"

from coreference import preprocess_coreference, extract_predicted_COREF_ids_from_output

document = {
    'text': "Marie a utilisé cet objet , un calculateur . Cette machine l' a beaucoup aidée",
    'mentions': [
        {'COREF': '0', 'onset': 0, 'offset': 5, 'text': 'Marie'},
        {'COREF': '0', 'onset': 16, 'offset': 25, 'text': 'cet objet'},
        {'COREF': "0", 'onset': 28, 'offset': 42, 'text': 'un calculateur'},
        {'COREF': "0", 'onset': 45, 'offset': 58, 'text': 'Cette machine'},
        {'COREF': '0', 'onset': 59, 'offset': 61, 'text': "l'"}
    ]
}

# adding the input field
document = preprocess_coreference(document)
print(document["input"])

messages = [{"role": "user", "content": PROMPT.replace("<input_placeholder>", document["input"])}]

prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
).replace("<think>\n\n</think>\n\n", "")
inputs = tokenizer.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

n_in = len(tokenizer.tokenizer(document["input"])["input_ids"])
out = model.generate(
    **inputs,
    max_new_tokens=max(int(1.25 * n_in), n_in + 20),
    do_sample=False,
    use_cache=True,
)
output = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
print(output)

predicted_COREF_ids = extract_predicted_COREF_ids_from_output(document["input"], output)
assert len(predicted_COREF_ids) == len(document["mentions"])

for mention, predicted_COREF in zip(document["mentions"], predicted_COREF_ids):
    mention["predicted_COREF"] = predicted_COREF
print(document["mentions"])
for cluster_ID in sorted(set([mention["predicted_COREF"] for mention in document["mentions"]])):
    print(f"Cluster ID: {cluster_ID}")
    cluster_mentions = [mention for mention in document["mentions"] if mention["predicted_COREF"] == cluster_ID]
    for mention in cluster_mentions:
        print(f"    {mention}")