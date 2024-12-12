import torch

def process_prompt(processor, qas, images=[], plans=[], locs=[], bos=True):
    conversations = []

    for q, a in qas:
        conversations.append(
            {
                "from": "human",
                "value": q,
            }
        )
        conversations.append(
            {
                "from": "gpt",
                "value": a,
            }
        )

    item = {"image": images, "conversations": conversations, "plan": plans, "loc": locs}

    _prompt = processor.process_item(item, bos=bos)
    prompt = []

    for value in _prompt:
        if isinstance(value, int):
            prompt.append(value)
        else:
            prompt += value["input_ids"]
    prompt = torch.tensor(prompt, dtype=torch.int64, device='cuda').unsqueeze(0)
    return prompt

def encode_mask(processor):
    mask_tokens = ''
    max_frames = 5
    # 3 plan tokens and 2 special tokens
    mask_len = 5 

    for i in range(max_frames):
        for j in range(mask_len):
            mask_tokens += f"<reserved{14696+i}>"
    
    return mask_tokens, process_prompt(processor, [[mask_tokens, None]], bos=False)

def decode_plans(plans):

    ps = plans.split('>')
    ts = list()

    for i, p in enumerate(ps):
        if i % 5 != 0 and i % 5 != 4:
            ts.append(int(p[-5:]))

    decoded = list()

    for i, t in enumerate(ts):
        if i % 3 == 0:
            decoded.append([])
        if i % 3 != 2:
            decoded[-1].append((t-10500)/50.-20)
        else:
            decoded[-1].append((t-15000)/100.-1.7)
    
    return decoded
