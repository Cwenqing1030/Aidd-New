"""Synthetic hypergraph + dynamics dataset generator.

Output format is intentionally aligned with the provided reference files:

1) Hypergraph structure file
   - one hyperedge per line
   - node ids are 0-based integers separated by a single space
   - e.g. a 3-node hyperedge: ``0 7 25``
   - when ``hyperedge_size=2``, this is exactly the same text layout as a
     normal graph edge-list file such as ``BA_N200_m2.txt``.

2) Dynamics X file
   - shape: [num_pairs, N]
   - row i is the state vector x(t_i)

3) Dynamics Y file
   - shape: [num_pairs, N]
   - row i is the next state vector x(t_i + 1)
   - therefore Y[i] is generated directly from X[i] by one dynamics step.

No header, commas, brackets, or row index are written, so np.loadtxt can read
all three files directly.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Tuple

import torch

from dynamics.contagion import contagion_F


def set_seed(seed: int) -> None:
    """Set the PyTorch random seed for reproducible topology and dynamics."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_fixed_size_incidence_matrix(
    N: int,
    E: int,
    hyperedge_size: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Generate a unique fixed-size hypergraph incidence matrix B in R^{N x E}.

    Rows are nodes and columns are hyperedges. B[i, e] = 1 means node i is a
    member of hyperedge e.
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if E <= 0:
        raise ValueError("E must be > 0")

    k = min(max(int(hyperedge_size), 1), N)
    max_edges = math.comb(N, k)
    if E > max_edges:
        raise ValueError(
            f"Cannot generate {E} unique hyperedges of size {k} with N={N}; "
            f"maximum is C({N},{k})={max_edges}."
        )

    B = torch.zeros((N, E), dtype=dtype, device=device)
    used = set()
    e = 0

    while e < E:
        # Sort node ids so the same hyperedge has one canonical representation.
        idx = torch.randperm(N, device=device)[:k]
        edge = tuple(sorted(idx.detach().cpu().tolist()))
        if edge in used:
            continue

        used.add(edge)
        B[list(edge), e] = 1.0
        e += 1

    return B


def incidence_to_hyperedges(B: torch.Tensor) -> list[tuple[int, ...]]:
    """Convert incidence matrix B [N, E] to a list of node-id tuples."""
    if B.ndim != 2:
        raise ValueError(f"B must be 2-D [N, E], got shape {tuple(B.shape)}")

    hyperedges: list[tuple[int, ...]] = []
    for e in range(B.shape[1]):
        nodes = torch.nonzero(B[:, e] != 0, as_tuple=False).flatten().tolist()
        if not nodes:
            raise ValueError(f"Hyperedge column {e} is empty")
        hyperedges.append(tuple(sorted(int(v) for v in nodes)))
    return hyperedges


def save_hypergraph_structure(B: torch.Tensor, path: str | Path) -> Path:
    """Save one hyperedge per line using space-separated 0-based node ids."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hyperedges = incidence_to_hyperedges(B)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for edge in hyperedges:
            f.write(" ".join(map(str, edge)) + "\n")

    return path


@torch.no_grad()
def generate_trajectory(
    B: torch.Tensor,
    num_pairs: int,
    beta: float,
    delta: float,
    dt: float,
    x0: Optional[torch.Tensor] = None,
    init_low: float = 0.0,
    init_high: float = 0.3,
) -> torch.Tensor:
    """Generate num_pairs + 1 consecutive states with shape [num_pairs+1, N].

    The returned rows are:
        trajectory[0] = x(0)
        trajectory[1] = x(1)
        ...
        trajectory[num_pairs] = x(num_pairs)
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")

    N = B.shape[0]
    if x0 is None:
        x = init_low + (init_high - init_low) * torch.rand(
            N, dtype=B.dtype, device=B.device
        )
    else:
        x = x0.to(dtype=B.dtype, device=B.device).flatten()
        if x.numel() != N:
            raise ValueError(f"x0 must contain N={N} values, got {x.numel()}")

    states = [x.clone()]
    for _ in range(num_pairs):
        x = contagion_F(x, B, beta=beta, delta=delta, dt=dt)
        x = x.to(dtype=B.dtype, device=B.device).flatten()
        if x.numel() != N:
            raise ValueError(
                f"contagion_F must return N={N} values, got shape {tuple(x.shape)}"
            )
        states.append(x.clone())

    return torch.stack(states, dim=0)  # [num_pairs + 1, N]


def make_xy_pairs(trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split consecutive states into X(T) and Y(T+1), both [samples, N]."""
    if trajectory.ndim != 2 or trajectory.shape[0] < 2:
        raise ValueError(
            "trajectory must have shape [T>=2, N], "
            f"got {tuple(trajectory.shape)}"
        )

    X = trajectory[:-1].contiguous()
    Y = trajectory[1:].contiguous()
    return X, Y


def _is_integer_valued(data: torch.Tensor, atol: float = 1e-12) -> bool:
    """Return True when every value is effectively an integer."""
    data = data.detach().cpu()
    return bool(torch.all(torch.isclose(data, torch.round(data), atol=atol, rtol=0.0)))


def save_dynamics_matrix(
    data: torch.Tensor,
    path: str | Path,
    float_precision: int = 10,
) -> Path:
    """Save [samples, N] states as whitespace-separated rows, without a header.

    Integer/binary tensors are written exactly like the supplied Ising files
    (e.g. ``1 0 1 ...``). Continuous tensors are written as plain decimals.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D [samples, N], got {tuple(data.shape)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_data = data.detach().cpu()

    integer_mode = _is_integer_valued(cpu_data)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        if integer_mode:
            values = torch.round(cpu_data).to(torch.int64).tolist()
            for row in values:
                f.write(" ".join(str(v) for v in row) + "\n")
        else:
            fmt = f"{{:.{int(float_precision)}g}}"
            for row in cpu_data.tolist():
                f.write(" ".join(fmt.format(float(v)) for v in row) + "\n")

    return path


def generate_and_save_dataset(
    output_dir: str | Path,
    prefix: str,
    N: int,
    E: int,
    hyperedge_size: int,
    num_pairs: int,
    beta: float,
    delta: float,
    dt: float,
    seed: int = 42,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    x0: Optional[torch.Tensor] = None,
) -> tuple[Path, Path, Path]:
    """Generate topology + X(T) + Y(T+1) and save the three txt files."""
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    B = generate_fixed_size_incidence_matrix(
        N=N,
        E=E,
        hyperedge_size=hyperedge_size,
        device=device,
        dtype=dtype,
    )

    trajectory = generate_trajectory(
        B=B,
        num_pairs=num_pairs,
        beta=beta,
        delta=delta,
        dt=dt,
        x0=x0,
    )
    X, Y = make_xy_pairs(trajectory)

    graph_path = output_dir / f"{prefix}.txt"
    x_path = output_dir / f"{prefix}_x.txt"
    y_path = output_dir / f"{prefix}_y.txt"

    save_hypergraph_structure(B, graph_path)
    save_dynamics_matrix(X, x_path)
    save_dynamics_matrix(Y, y_path)

    # Sanity checks: X/Y must be aligned exactly as T -> T+1 pairs.
    assert X.shape == (num_pairs, N)
    assert Y.shape == (num_pairs, N)
    assert torch.equal(X[1:], Y[:-1]), "Temporal alignment check failed"

    print("Generation complete")
    print(f"B shape: {tuple(B.shape)}  (incidence matrix, not written directly)")
    print(f"X shape: {tuple(X.shape)}  -> {x_path}")
    print(f"Y shape: {tuple(Y.shape)}  -> {y_path}")
    print(f"Hyperedges: {E}  -> {graph_path}")
    print("Temporal check: Y[i] = X[i+1] for consecutive rows: PASS")

    return graph_path, x_path, y_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a fixed-size hypergraph and aligned X(T)/Y(T+1) dynamics files."
    )
    parser.add_argument("--output-dir", type=str, default="generated_data")
    parser.add_argument("--prefix", type=str, default="hypergraph_N200")
    parser.add_argument("--N", type=int, default=200, help="number of nodes")
    parser.add_argument("--E", type=int, default=400, help="number of hyperedges")
    parser.add_argument(
        "--hyperedge-size", type=int, default=3, help="nodes per hyperedge"
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=4999,
        help="number of X(T), Y(T+1) sample pairs",
    )
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    generate_and_save_dataset(
        output_dir=args.output_dir,
        prefix=args.prefix,
        N=args.N,
        E=args.E,
        hyperedge_size=args.hyperedge_size,
        num_pairs=args.num_pairs,
        beta=args.beta,
        delta=args.delta,
        dt=args.dt,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
