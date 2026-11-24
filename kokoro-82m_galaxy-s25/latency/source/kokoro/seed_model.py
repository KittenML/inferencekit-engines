"""Pin the existing Kokoro vocoder's per-session RNG streams for paired tests."""
import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnx import AttributeProto


SEEDS = {
    "/decoder/generator/m_source/l_sin_gen/RandomUniformLike": ("RandomUniformLike", 8201),
    "/decoder/generator/m_source/l_sin_gen/RandomNormalLike": ("RandomNormalLike", 8209),
}


def nodes(graph):
    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                yield from nodes(attribute.g)
            elif attribute.type == AttributeProto.GRAPHS:
                for nested in attribute.graphs:
                    yield from nodes(nested)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    model = onnx.load(args.source, load_external_data=False)
    random_nodes = [node for node in nodes(model.graph)
                    if node.op_type.startswith("Random") or node.op_type in {"Bernoulli", "Multinomial"}]
    if len(random_nodes) != len(SEEDS) or {node.name for node in random_nodes} != SEEDS.keys():
        raise ValueError("Unexpected Kokoro random-node inventory")
    original = hashlib.sha256(model.SerializeToString(deterministic=True)).hexdigest()
    for node in random_nodes:
        operation, seed = SEEDS[node.name]
        if node.domain or node.op_type != operation or any(a.name == "seed" for a in node.attribute):
            raise ValueError(f"Unexpected operation or existing seed for {node.name}")
        node.attribute.append(onnx.helper.make_attribute("seed", float(seed)))
    temporary = args.output.with_suffix(".onnx.partial")
    onnx.save_model(model, temporary)
    for node in random_nodes:
        del node.attribute[-1]
    without_seeds = hashlib.sha256(model.SerializeToString(deterministic=True)).hexdigest()
    if without_seeds != original:
        raise ValueError("Normalized graph changed beyond the two seed attributes")
    del model
    onnx.checker.check_model(str(temporary))
    temporary.replace(args.output)
    args.evidence.write_text(json.dumps({
        "source_sha256": digest(args.source),
        "seeded_sha256": digest(args.output),
        "normalized_source_sha256": original,
        "normalization_removes_only_seed_attributes": True,
        "onnx_version": onnx.__version__,
        "seeds": {name: seed for name, (_, seed) in SEEDS.items()},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
