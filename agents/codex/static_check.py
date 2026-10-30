"""Source-only audit. Never imports or executes the contestant, torch, or maze_game."""

import ast
import json
import sys
from pathlib import Path


def first_statement(function):
    statements = list(function.body)
    if statements and isinstance(statements[0], ast.Expr) and isinstance(statements[0].value, ast.Constant):
        statements = statements[1:]
    return statements[0] if statements else None


def guarded(function) -> bool:
    statement = first_statement(function)
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "require_ready"
    )


def main() -> int:
    root = Path(__file__).resolve().parent
    sources = {
        path.stem: (path, ast.parse(path.read_text(), filename=str(path))) for path in root.glob("*.py")
    }
    errors = []
    for name, (path, tree) in sources.items():
        compile(tree, str(path), "exec")  # Compilation only. This code object is never executed.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("maze_game._"):
                errors.append(f"{name}:{node.lineno}: private game import")
            if isinstance(node, ast.Attribute) and node.attr in {
                "_maze",
                "_layout",
                "_optimal_moves",
                "_seed",
                "_move_limit",
                "_shortest_path",
            }:
                errors.append(f"{name}:{node.lineno}: hidden environment attribute")
        for statement in tree.body:
            # Only definitions, imports, literal assignments, and __main__ guards
            # belong at module scope. No top-level session/model/IO calls.
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                errors.append(f"{name}:{statement.lineno}: top-level executable call")
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                try:
                    ast.literal_eval(statement.value)
                except (ValueError, TypeError):
                    errors.append(f"{name}:{statement.lineno}: nonliteral module initialization")
    ready = [
        node
        for node in sources["activation"][1].body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "READY" for target in node.targets)
    ]
    if len(ready) != 1 or not isinstance(ready[0].value, ast.Constant) or ready[0].value.value is not False:
        errors.append("activation.py must contain exactly one READY = False assignment for dormant delivery")
    classes = {
        "agent": ("Policy", "CodexAgent"),
        "memory": ("SpatialMemory",),
        "network": ("Encoder", "Navigator"),
        "curriculum": ("Curriculum",),
        "trainer": ("Slot", "Trainer"),
        "recording": ("StudioRecorder",),
    }
    guard_count = 0
    for module, names in classes.items():
        for name in names:
            definition = next(
                item
                for item in sources[module][1].body
                if isinstance(item, ast.ClassDef) and item.name == name
            )
            for method in definition.body:
                if isinstance(method, ast.FunctionDef) and method.name not in ("ready", "seconds"):
                    guard_count += 1
                    if not guarded(method):
                        errors.append(
                            f"{module}.{name}.{method.name}: first operation must require readiness"
                        )
    functions = {
        "runtime": (
            "configure",
            "select_device",
            "save_checkpoint",
            "read_checkpoint",
            "restore_rng",
            "append_json",
        ),
        "ppo": ("advantages", "optimize"),
        "evaluation": ("evaluate_policy", "summarize", "run_evaluation"),
        "network": ("tensor_inputs", "sample_action"),
        "trainer": ("train",),
        "curriculum": ("maze_seed",),
    }
    for module, names in functions.items():
        for name in names:
            definition = next(
                item
                for item in sources[module][1].body
                if isinstance(item, ast.FunctionDef) and item.name == name
            )
            guard_count += 1
            if not guarded(definition):
                errors.append(f"{module}.{name}: first operation must require readiness")
    # Check relative import targets and explicitly imported symbols from local
    # modules using their AST definitions, never by importing those modules.
    for module, (_, tree) in sources.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
                continue
            target = sources.get(node.module)
            if target is None:
                errors.append(f"{module}: unresolved relative module {node.module}")
                continue
            symbols = {
                item.name for item in target[1].body if isinstance(item, (ast.ClassDef, ast.FunctionDef))
            }
            for item in target[1].body:
                if isinstance(item, ast.Assign):
                    symbols.update(t.id for t in item.targets if isinstance(t, ast.Name))
            for item in node.names:
                if item.name not in symbols:
                    errors.append(f"{module}: unresolved symbol {node.module}.{item.name}")
    config_tree = sources["config"][1]
    configs = {item.name: item for item in config_tree.body if isinstance(item, ast.ClassDef)}
    allowed = {item.target.id for item in configs["Config"].body if isinstance(item, ast.AnnAssign)}
    for path in (root / "configs").glob("*.json"):
        values = json.loads(path.read_text())
        if not set(values) <= allowed:
            errors.append(f"{path.name}: unknown config keys")
        if values.get("rollout_steps", 256) % values.get("sequence_length", 32):
            errors.append(f"{path.name}: recurrent sequence length mismatch")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        f"Static audit: {len(sources)} Python files parsed/compiled; {guard_count} guarded methods/functions checked."
    )
    print("READY is literally False. No contestant module, model, environment, or episode was executed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
