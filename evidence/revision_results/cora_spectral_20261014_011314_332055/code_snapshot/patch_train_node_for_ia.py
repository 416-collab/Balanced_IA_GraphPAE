from pathlib import Path

p = Path("train_node.py")
text = p.read_text()

backup = Path("train_node_before_ia_budget_patch.py")
if not backup.exists():
    backup.write_text(text)
    print("Backup saved:", backup)
else:
    print("Backup already exists:", backup)

if "from ia_utils import select_budgeted_pe_edges" not in text:
    if "import torch\n" in text:
        text = text.replace("import torch\n", "import torch\nfrom ia_utils import select_budgeted_pe_edges\n", 1)
    else:
        text = "from ia_utils import select_budgeted_pe_edges\n" + text

lines = text.splitlines(keepends=True)

start = None
for i, line in enumerate(lines):
    if line.startswith("def train_mae_epoch("):
        start = i
        break

if start is None:
    raise RuntimeError("Could not find def train_mae_epoch in train_node.py")

end = len(lines)
for j in range(start + 1, len(lines)):
    line = lines[j]
    if line.startswith("def ") or line.startswith("class ") or line.startswith("if __name__"):
        end = j
        break

new_func_lines = [
"def train_mae_epoch(graph_auto_encoder, x, edge_index, edge_index_pe, u, PE, optimizer):\n",
"    graph_auto_encoder.train()\n",
"\n",
"    run_args = getattr(graph_auto_encoder, 'args', None)\n",
"    if run_args is None:\n",
"        run_args = globals().get('args', None)\n",
"\n",
"    edge_index_pe_epoch = select_budgeted_pe_edges(\n",
"        edge_index_pe=edge_index_pe,\n",
"        u=u,\n",
"        graph_edge_index=edge_index,\n",
"        args=run_args,\n",
"        device=x.device,\n",
"    )\n",
"\n",
"    optimizer.zero_grad(set_to_none=True)\n",
"\n",
"    use_amp = bool(getattr(run_args, 'use_amp', False)) if run_args is not None else False\n",
"    amp_dtype_name = getattr(run_args, 'amp_dtype', 'bf16') if run_args is not None else 'bf16'\n",
"    amp_dtype = torch.bfloat16 if amp_dtype_name == 'bf16' else torch.float16\n",
"\n",
"    with torch.autocast(\n",
"        device_type='cuda',\n",
"        dtype=amp_dtype,\n",
"        enabled=(use_amp and x.device.type == 'cuda'),\n",
"    ):\n",
"        loss = graph_auto_encoder(x, edge_index, u, PE, edge_index_pe_epoch)\n",
"\n",
"    loss.backward()\n",
"    optimizer.step()\n",
"    return loss.detach()\n",
"\n",
]

lines = lines[:start] + new_func_lines + lines[end:]
p.write_text("".join(lines))

print("train_node.py patched successfully.")
print("Replaced train_mae_epoch lines:", start + 1, "to", end)
