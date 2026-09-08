import argparse
from pathlib import Path

import onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a fixed-height/fixed-width simplified ONNX model.")
    parser.add_argument("--model", required=True, type=Path, help="Source ONNX model")
    parser.add_argument("--output", required=True, type=Path, help="Output ONNX model")
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    args = parser.parse_args()
    if args.height < 1 or args.width < 1:
        parser.error("--height and --width must be positive")
    if not args.model.is_file():
        parser.error(f"Model does not exist: {args.model}")
    return args


def set_input_shape(model: onnx.ModelProto, height: int, width: int) -> str:
    if len(model.graph.input) != 1:
        raise RuntimeError(f"Expected exactly one model input, found {len(model.graph.input)}")
    input_value = model.graph.input[0]
    dimensions = input_value.type.tensor_type.shape.dim
    if len(dimensions) != 3:
        raise RuntimeError(f"Expected rank-3 input, found rank {len(dimensions)}")
    dimensions[0].ClearField("dim_param")
    dimensions[0].dim_value = height
    dimensions[1].ClearField("dim_param")
    dimensions[1].dim_value = width
    if not dimensions[2].HasField("dim_value") or dimensions[2].dim_value != 1:
        raise RuntimeError("Expected a fixed single-channel input")
    return input_value.name


def main() -> None:
    args = parse_args()
    model = onnx.load(args.model)
    input_name = set_input_shape(model, args.height, args.width)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)

    try:
        from onnxsim import simplify
    except ImportError as exc:
        raise RuntimeError("onnxsim is required; install it with: pip install onnxsim") from exc

    simplified, check_ok = simplify(model, input_shapes={input_name: [args.height, args.width, 1]})
    if not check_ok:
        raise RuntimeError("onnxsim validation failed")
    onnx.checker.check_model(simplified)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(simplified, args.output)
    print(f"[DONE] Wrote {args.output}")
    print(f"[INFO] Input shape: {args.height}x{args.width}x1")
    print(f"[INFO] Nodes: {len(model.graph.node)} -> {len(simplified.graph.node)}")
    print(f"[INFO] Initializers: {len(model.graph.initializer)} -> {len(simplified.graph.initializer)}")


if __name__ == "__main__":
    main()
