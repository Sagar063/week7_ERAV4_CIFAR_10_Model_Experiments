# rf_utils.py
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class LayerSpec:
    name: str
    k: int       # kernel size (assume square)
    stride: int  # stride
    dilation: int  # dilation

def rf_trace(layers: List[LayerSpec], rf_start: int = 1, jump_start: int = 1) -> List[dict]:
    """
    Compute layer-by-layer Receptive Field (RF) and 'jump' (a.k.a. effective stride).
    Assumes same padding keeps spatial dims (which your code does).
    For each layer:
      rf_out = rf_in + (k - 1) * dilation * jump_in
      jump_out = jump_in * stride
    """
    rf = rf_start
    jump = jump_start
    out = []
    for i, L in enumerate(layers, 1):
        rf_next  = rf + (L.k - 1) * L.dilation * jump
        jump_next = jump * L.stride
        out.append({
            "idx": i,
            "name": L.name,
            "k": L.k,
            "stride": L.stride,
            "dilation": L.dilation,
            "rf_in": rf,
            "jump_in": jump,
            "rf_out": rf_next,
            "jump_out": jump_next,
        })
        rf, jump = rf_next, jump_next
    return out

def rf_markdown_table(trace: List[dict], title: str) -> str:
    lines = [f"### {title}", "", 
             "| # | Layer | k | s | d | RF_in | jump_in | RF_out | jump_out |",
             "|---:|:------|:-:|:-:|:-:|-----:|--------:|-------:|---------:|"]
    for row in trace:
        lines.append(f"| {row['idx']} | {row['name']} | {row['k']} | {row['stride']} | {row['dilation']} | "
                     f"{row['rf_in']} | {row['jump_in']} | {row['rf_out']} | {row['jump_out']} |")
    # Final summary line
    if trace:
        lines += ["", f"**Final RF:** {trace[-1]['rf_out']} &nbsp;&nbsp; **Final jump:** {trace[-1]['jump_out']}"]
    lines.append("")  # trailing newline
    return "\n".join(lines)
