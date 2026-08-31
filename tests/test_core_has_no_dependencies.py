"""Enforces the layering claim in recoup/__init__.py.

The claim is that `domain`, `audit`, `diagnosis` and `policy` import only the
standard library, and that no inner layer imports an outer one. Both halves are
easy to break by accident — one convenient `import numpy` inside a policy rule
for a percentile, one `from recoup.gateway import ...` for a type annotation —
and neither breaks any other test. So they are checked directly, by parsing the
source rather than by importing it.

Parsing rather than importing matters. A test that imports the modules and
inspects `sys.modules` would pass on a machine where numpy happens to be absent
and the offending import sits inside a function, and would also be confounded by
whatever the test runner itself has already imported. The AST does not care what
is installed.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "recoup"

# Layers in dependency order. A module in one layer may import its own layer or
# any layer above it, never below.
#
# `simulation` sits below `gateway` deliberately, and the first version of this
# list had it above. The simulation is not a lower-level utility that the gateway
# builds on; it is a harness that drives the gateway, the policy engine and the
# model together to produce the evaluation numbers. Putting it above `gateway`
# made this test fail on three files that were all correct.
LAYERS: tuple[str, ...] = (
    "domain",
    "audit",
    "diagnosis",
    "policy",
    "model",
    "gateway",
    "simulation",
    "api",
)

# The layers whose only permitted third-party dependency count is zero.
STDLIB_ONLY: frozenset[str] = frozenset({"domain", "audit", "diagnosis", "policy"})


def _modules(layer: str) -> list[pathlib.Path]:
    directory = PACKAGE / layer
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*.py") if "__pycache__" not in path.parts)


def _imported_roots(path: pathlib.Path) -> set[str]:
    """Top-level package names imported anywhere in the file, including inside functions.

    Relative imports are resolved to `recoup.<layer>` so they are checked by the
    same rule as absolute ones; a layer violation hidden behind `from ..gateway
    import X` is still a layer violation.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
                base = parts[: len(parts) - node.level]
                target = ".".join((*base, node.module) if node.module else base)
                roots.add(target.split(".")[0] if not target.startswith("recoup") else "recoup")
                if target.startswith("recoup."):
                    roots.add(target)
            else:
                roots.add(node.module.split(".")[0] if node.module else "")
                if node.module and node.module.startswith("recoup."):
                    roots.add(node.module)
    return roots


def _recoup_layers_used(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    used: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "recoup" and len(parts) > 1:
                used.add(parts[1])
    return used


class StdlibOnlyTests(unittest.TestCase):
    def test_core_layers_import_only_the_standard_library(self):
        stdlib = sys.stdlib_module_names
        offenders: list[str] = []
        for layer in sorted(STDLIB_ONLY):
            for path in _modules(layer):
                for root in _imported_roots(path):
                    if not root or root.startswith("recoup"):
                        continue
                    if root not in stdlib:
                        offenders.append(f"{path.relative_to(ROOT)} imports {root}")
        self.assertEqual(
            offenders,
            [],
            "the four core layers must run with nothing installed:\n  "
            + "\n  ".join(offenders),
        )

    def test_the_claim_covers_every_core_layer_that_exists(self):
        """Guards against the check passing because a directory was renamed away."""
        for layer in STDLIB_ONLY:
            self.assertTrue(_modules(layer), f"recoup/{layer} has no modules to check")


class LayeringTests(unittest.TestCase):
    def test_no_layer_imports_one_below_it(self):
        rank = {layer: index for index, layer in enumerate(LAYERS)}
        offenders: list[str] = []
        for layer in LAYERS:
            for path in _modules(layer):
                for used in _recoup_layers_used(path):
                    if used not in rank:
                        offenders.append(f"{path.relative_to(ROOT)} imports unknown layer {used}")
                    elif rank[used] > rank[layer]:
                        offenders.append(
                            f"{path.relative_to(ROOT)} imports recoup.{used}, "
                            f"which is below recoup.{layer}"
                        )
        self.assertEqual(
            offenders, [], "layering violations:\n  " + "\n  ".join(offenders)
        )

    def test_the_policy_engine_does_not_know_about_the_gateway(self):
        """Stated separately because it is the boundary that makes shadow mode work.

        `decide()` is pure: it returns a decision and executes nothing. If the
        policy layer could reach the gateway, someone would eventually make it
        execute, and shadow mode — running the agent over historical failures with
        no side effects — would stop being possible without a mocking framework.
        """
        for path in _modules("policy"):
            self.assertNotIn(
                "gateway",
                _recoup_layers_used(path),
                f"{path.relative_to(ROOT)} reaches into the gateway layer",
            )


class OptionalDependencyTests(unittest.TestCase):
    def test_numpy_is_confined_to_the_model_and_simulation_layers(self):
        """numpy may be absent, so anything importing it at module scope is optional.

        `recoup.model.__init__` is exempt: its whole job is to import the numeric
        modules lazily and turn the ImportError into a message that names the
        extra to install.
        """
        allowed = {"model", "simulation"}
        for layer in LAYERS:
            if layer in allowed:
                continue
            for path in _modules(layer):
                self.assertNotIn(
                    "numpy",
                    _imported_roots(path),
                    f"{path.relative_to(ROOT)} imports numpy outside the numeric layers",
                )


if __name__ == "__main__":
    unittest.main()
