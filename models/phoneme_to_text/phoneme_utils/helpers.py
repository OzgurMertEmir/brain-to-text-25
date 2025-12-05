import json

def get_raw_phonemes():
    with open("raw_phonemes.json", "r") as f:
        phonemes = json.load(f)
        return phonemes
        
def get_raw_diphones():
    with open("raw_diphones.json", "r") as f:
        diphones = json.load(f)
        return diphones

def get_raw_triphones():
    with open("raw_triphones.json", "r") as f:
        triphones = json.load(f)
        return triphones

def get_phoneme_tokens(type_, ph_seq):
    tag = None
    if type_ == "mono":
        tag = "p"
    elif type_ == "diphone":
        tag = "d"
        ph_seq = _make_diphone_tokens(ph_seq)
    elif type_ == "triphone":
        tag = "t"
        ph_seq = _make_triphone_tokens(ph_seq)
    
    PHONEME_MAP = {p: f"<{tag}:{p.strip()}>" for p in ph_seq}
    PHONEME_TOKENS = list(PHONEME_MAP.values())
    return PHONEME_TOKENS

BOUNDARY_SYMBOL = "|"
def _make_diphone_tokens(ph_seq):
    """
    Example:
        ph_seq = ['HH', 'EH', 'L', 'OW']
        -> [' | ', 'HH', 'EH', 'L', 'OW', ' | ']
        -> ['<d:|->HH>', '<d:HH->EH>', ..., '<d:OW->|>']
    """
    ph_seq = list(ph_seq)
    if ph_seq[0] != BOUNDARY_SYMBOL:
        ph_seq.insert(0, BOUNDARY_SYMBOL)
    if ph_seq[-1] != BOUNDARY_SYMBOL:
        ph_seq.append(BOUNDARY_SYMBOL)
    toks = []
    for i in range(len(ph_seq) - 1):
        a = ph_seq[i]
        b = ph_seq[i + 1]
        toks.append(f"{a.strip()}-{b.strip()}")
    return toks

def _make_triphone_tokens(ph_seq):
    """
    Example:
        ph_seq = ['HH', 'EH', 'L', 'OW']
        seq_ext = [' | ', 'HH', 'EH', 'L', 'OW', ' | ']
        -> i=1..4:
            '<t:|-HH-EH>', '<t:HH-EH-L>', '<t:EH-L-OW>', '<t:L-OW-|>'
    """
    ph_seq = list(ph_seq)
    if ph_seq[0] != BOUNDARY_SYMBOL:
        ph_seq.insert(0, BOUNDARY_SYMBOL)
    if ph_seq[-1] != BOUNDARY_SYMBOL:
        ph_seq.append(BOUNDARY_SYMBOL)
    toks = []
    for i in range(1, len(ph_seq) - 1):
        a = ph_seq[i - 1]
        b = ph_seq[i]
        c = ph_seq[i + 1]
        toks.append(f"{a.strip()}-{b.strip()}-{c.strip()}")
    return toks