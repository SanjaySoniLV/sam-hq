"""Decompose LayerNormalization in encoder ONNX to primitive ops for DirectML/ORT.

See export workflow and LayerNorm Decomposition report for context.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _attr_value(node: onnx.NodeProto, name: str, default):
    for attr in node.attribute:
        if attr.name != name:
            continue
        if attr.type == onnx.AttributeProto.INT:
            return int(attr.i)
        if attr.type == onnx.AttributeProto.FLOAT:
            return float(attr.f)
    return default


def rewrite_layernorm_encoder(src: Path, dst: Path) -> dict[str, int]:
    model = onnx.load(str(src), load_external_data=True)
    graph = model.graph

    new_nodes: list[onnx.NodeProto] = []
    added_inits: list[onnx.TensorProto] = []

    eps_cache: dict[float, str] = {}
    axes_cache: dict[int, str] = {}
    one_name = "__ln_decomp_const_one_f32"
    one_added = False

    rewritten = 0

    for node in graph.node:
        if node.op_type != "LayerNormalization":
            new_nodes.append(copy.deepcopy(node))
            continue

        rewritten += 1

        node_name = node.name or f"LayerNormalization_{rewritten}"
        prefix = f"{node_name}__decomp"

        x = node.input[0]
        scale = node.input[1] if len(node.input) > 1 and node.input[1] else ""
        bias = node.input[2] if len(node.input) > 2 and node.input[2] else ""

        y_out = node.output[0]
        mean_out = node.output[1] if len(node.output) > 1 and node.output[1] else f"{prefix}/mean"
        invstd_out = node.output[2] if len(node.output) > 2 and node.output[2] else f"{prefix}/invstd"

        axis = _attr_value(node, "axis", -1)
        epsilon = float(_attr_value(node, "epsilon", 1e-5))

        if axis not in axes_cache:
            axes_cache[axis] = f"axis_{axis}"

        if epsilon not in eps_cache:
            eps_name = f"__ln_decomp_eps_{len(eps_cache)}"
            eps_cache[epsilon] = eps_name
            added_inits.append(
                numpy_helper.from_array(np.array(epsilon, dtype=np.float32), name=eps_name)
            )
        eps_name = eps_cache[epsilon]

        if not one_added and len(node.output) > 2 and node.output[2]:
            added_inits.append(
                numpy_helper.from_array(np.array(1.0, dtype=np.float32), name=one_name)
            )
            one_added = True

        centered = f"{prefix}/centered"
        sq = f"{prefix}/sq"
        var = f"{prefix}/var"
        var_eps = f"{prefix}/var_eps"
        std = f"{prefix}/std"
        norm = f"{prefix}/norm"
        scaled = f"{prefix}/scaled"

        new_nodes.append(
            helper.make_node(
                "ReduceMean",
                inputs=[x],
                outputs=[mean_out],
                axes=[axis],
                keepdims=1,
                name=f"{prefix}/ReduceMean",
            )
        )
        new_nodes.append(
            helper.make_node(
                "Sub",
                inputs=[x, mean_out],
                outputs=[centered],
                name=f"{prefix}/Sub",
            )
        )
        new_nodes.append(
            helper.make_node(
                "Mul",
                inputs=[centered, centered],
                outputs=[sq],
                name=f"{prefix}/MulSquare",
            )
        )
        new_nodes.append(
            helper.make_node(
                "ReduceMean",
                inputs=[sq],
                outputs=[var],
                axes=[axis],
                keepdims=1,
                name=f"{prefix}/ReduceMeanVar",
            )
        )
        new_nodes.append(
            helper.make_node(
                "Add",
                inputs=[var, eps_name],
                outputs=[var_eps],
                name=f"{prefix}/AddEps",
            )
        )
        new_nodes.append(
            helper.make_node(
                "Sqrt",
                inputs=[var_eps],
                outputs=[std],
                name=f"{prefix}/Sqrt",
            )
        )
        new_nodes.append(
            helper.make_node(
                "Div",
                inputs=[centered, std],
                outputs=[norm],
                name=f"{prefix}/DivNormalize",
            )
        )

        if scale:
            new_nodes.append(
                helper.make_node(
                    "Mul",
                    inputs=[norm, scale],
                    outputs=[scaled],
                    name=f"{prefix}/MulScale",
                )
            )
            affine = scaled
        else:
            affine = norm

        if bias:
            new_nodes.append(
                helper.make_node(
                    "Add",
                    inputs=[affine, bias],
                    outputs=[y_out],
                    name=f"{prefix}/AddBias",
                )
            )
        else:
            new_nodes.append(
                helper.make_node(
                    "Identity",
                    inputs=[affine],
                    outputs=[y_out],
                    name=f"{prefix}/IdentityOut",
                )
            )

        if len(node.output) > 2 and node.output[2]:
            new_nodes.append(
                helper.make_node(
                    "Div",
                    inputs=[one_name, std],
                    outputs=[invstd_out],
                    name=f"{prefix}/DivInvStd",
                )
            )

    del graph.node[:]
    graph.node.extend(new_nodes)

    existing_init_names = {t.name for t in graph.initializer}
    for init in added_inits:
        if init.name not in existing_init_names:
            graph.initializer.append(init)
            existing_init_names.add(init.name)

    onnx.checker.check_model(model)
    onnx.save_model(model, str(dst))

    return {
        "rewritten_layernorm_nodes": rewritten,
        "total_nodes_after_rewrite": len(graph.node),
        "added_initializers": len(added_inits),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace LayerNormalization nodes in a SAM-HQ encoder ONNX with primitive ops."
    )
    parser.add_argument("src", type=Path, help="Input encoder ONNX path.")
    parser.add_argument("dst", type=Path, help="Output encoder ONNX path.")
    args = parser.parse_args()
    stats = rewrite_layernorm_encoder(args.src, args.dst)
    print(f"Wrote patched encoder: {args.dst}")
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
