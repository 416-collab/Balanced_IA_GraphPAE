from pathlib import Path
import yaml

CONFIG = Path("config/arxiv-year-official.yaml")

def write_config(layers, dim, heads, pe_edges, name):
    c = yaml.safe_load(CONFIG.read_text())

    c["device"] = 0
    c["feat_dim"] = 1

    # quick OOM test only
    c["epochs"] = 2
    c["epochs_eval"] = 20

    c["enc_gnn_layer"] = layers
    c["embed_dim"] = dim
    c["heads"] = heads

    c["max_freqs"] = 100
    c["max_pe_edges"] = pe_edges

    c["pe_loss_mode"] = "random"
    c["pe_weight_mode"] = "none"

    c["masked_pe_loss"] = 0.05
    c["masked_atom_loss"] = 1.0
    c["mask_ratio"] = 0.5
    c["replace_ratio"] = 0.0

    c["gnn_dropout"] = 0.3
    c["gnn_edge_dropout"] = 0.5

    c["use_amp"] = True
    c["amp_dtype"] = "bf16"
    c["use_checkpoint"] = True

    c["atom_recon_type"] = "sce"
    c["alpha_l"] = 2.0
    c["noise_val"] = 0.01

    c["optim"] = "adam"
    c["init_lr"] = 0.001
    c["weight_decay"] = 0.0
    c["use_schedule"] = True
    c["step_size"] = 50
    c["gamma"] = 0.5

    c["lr_eval"] = 0.01
    c["wd_eval"] = 0.0001

    c["gnn_activation"] = "elu"
    c["task_head_activation"] = "gelu"
    c["pos_activation"] = "gelu"

    CONFIG.write_text(yaml.safe_dump(c, sort_keys=False))

    print("=" * 80)
    print("TEST:", name)
    print("layers:", layers, "dim:", dim, "heads:", heads, "pe_edges:", pe_edges)
    print("=" * 80)

if __name__ == "__main__":
    import sys
    name = sys.argv[1]

    tests = {
        "T1_3L_160H1_300k":  (3, 160, 1, 300000),
        "T2_3L_160H1_500k":  (3, 160, 1, 500000),

        "T3_2L_256H1_500k":  (2, 256, 1, 500000),
        "T4_2L_256H2_300k":  (2, 256, 2, 300000),
        "T5_3L_256H1_300k":  (3, 256, 1, 300000),

        "T6_2L_512H1_200k":  (2, 512, 1, 200000),
        "T7_2L_512H2_200k":  (2, 512, 2, 200000),
        "T8_2L_512H4_100k":  (2, 512, 4, 100000),

        "T9_3L_512H1_100k":  (3, 512, 1, 100000),
        "T10_3L_512H2_100k": (3, 512, 2, 100000),

        # very risky, close to paper style but reduced PE edges
        "T11_2L_1024H4_50k": (2, 1024, 4, 50000),

        # boundary tests after 512 OOM
        "T12_2L_384H1_200k": (2, 384, 1, 200000),
        "T13_2L_384H1_300k": (2, 384, 1, 300000),
        "T14_2L_384H2_100k": (2, 384, 2, 100000),
        "T15_3L_384H1_100k": (3, 384, 1, 100000),
        "T16_2L_448H1_100k": (2, 448, 1, 100000),
    }

    if name not in tests:
        print("Available tests:")
        for k in tests:
            print(k)
        raise SystemExit(1)

    write_config(*tests[name], name=name)
