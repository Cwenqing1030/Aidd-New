"""Generate a hypergraph SIR dataset for the AIDD/CoND-style training pipeline.

Core output (one dataset directory):

1) <prefix>.txt
   Hypergraph structure: one hyperedge per line, 0-based node ids separated by spaces.
   Example for a 3-uniform hypergraph::

       0 7 25
       3 19 41

2) <prefix>_SIR_x.txt
   Shape [num_pairs, N].  Each entry is a discrete SIR state:
       0 = Susceptible (S), 1 = Infected (I), 2 = Recovered (R)

3) <prefix>_SIR_y.txt
   Shape [num_pairs, N].  Row r is exactly the one-step SIR successor of
   row r in X.

The naming follows the loader convention used by dynamics_predict.py:
    <network_name>.txt
    <network_name>_<dynamics_type>_x.txt
    <network_name>_<dynamics_type>_y.txt

Important:
- This file generates a *true hypergraph* incidence structure B in R^{N x E}.
- The uploaded graph-only dynamics_predict.py loader accepts only an NxN adjacency
  matrix or a two-column graph edge list.  A genuine k>=3 hyperedge list therefore
  requires the hypergraph/NxE loader version of AIDD.  The SIR X/Y files themselves
  already follow the same discrete-state format expected by NN3/CrossEntropyLoss.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import torch


S_STATE = 0
I_STATE = 1
R_STATE = 2


def set_seed(seed: int) -> None:
    """Set PyTorch random seeds for reproducible topology and trajectories."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_fixed_size_incidence_matrix(
    N: int,
    E: int,
    hyperedge_size: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a unique k-uniform hypergraph incidence matrix B [N, E].

    Rows are nodes and columns are hyperedges.  B[i, e] = 1 means node i belongs
    to hyperedge e.
    """
    if N <= 0:
        raise ValueError("N must be > 0")
    if E <= 0:
        raise ValueError("E must be > 0")
    if not (1 <= hyperedge_size <= N):
        raise ValueError("hyperedge_size must satisfy 1 <= hyperedge_size <= N")

    max_edges = math.comb(N, hyperedge_size)
    if E > max_edges:
        raise ValueError(
            f"Cannot generate {E} unique hyperedges of size {hyperedge_size} "
            f"with N={N}; maximum is C({N},{hyperedge_size})={max_edges}."
        )

    B = torch.zeros((N, E), dtype=torch.float32, device=device)
    used: set[tuple[int, ...]] = set()
    e = 0

    while e < E:
        nodes = torch.randperm(N, device=device)[:hyperedge_size]
        edge = tuple(sorted(int(v) for v in nodes.detach().cpu().tolist()))
        if edge in used:
            continue
        used.add(edge)
        B[list(edge), e] = 1.0
        e += 1

    return B


def incidence_to_hyperedges(B: torch.Tensor) -> list[tuple[int, ...]]:
    """Convert incidence matrix B [N,E] to a list of hyperedge node tuples."""
    if B.ndim != 2:
        raise ValueError(f"B must be 2-D [N,E], got {tuple(B.shape)}")

    edges: list[tuple[int, ...]] = []
    for e in range(B.shape[1]):
        nodes = torch.nonzero(B[:, e] > 0, as_tuple=False).flatten().tolist()
        if len(nodes) == 0:
            raise ValueError(f"Hyperedge {e} is empty")
        edges.append(tuple(sorted(int(v) for v in nodes)))
    return edges


def save_hypergraph_structure(B: torch.Tensor, path: str | Path) -> Path:
    """Write one hyperedge per line with 0-based node ids."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for edge in incidence_to_hyperedges(B):
            f.write(" ".join(map(str, edge)) + "\n")
    return path


def initialize_sir_state(
    N: int,
    infected_fraction: float,
    recovered_fraction: float,
    device: str,
) -> torch.Tensor:
    """Create one initial SIR state vector [N] using labels 0/1/2."""
    if not (0.0 < infected_fraction < 1.0):
        raise ValueError("infected_fraction must be in (0,1)")
    if not (0.0 <= recovered_fraction < 1.0):
        raise ValueError("recovered_fraction must be in [0,1)")
    if infected_fraction + recovered_fraction >= 1.0:
        raise ValueError("infected_fraction + recovered_fraction must be < 1")

    n_infected = max(1, int(round(N * infected_fraction)))
    n_recovered = int(round(N * recovered_fraction))
    if n_infected + n_recovered >= N:
        n_recovered = max(0, N - n_infected - 1)

    x = torch.full((N,), S_STATE, dtype=torch.long, device=device)
    perm = torch.randperm(N, device=device)
    x[perm[:n_infected]] = I_STATE
    if n_recovered > 0:
        x[perm[n_infected : n_infected + n_recovered]] = R_STATE
    return x


@torch.no_grad()
def hypergraph_sir_step(
    x: torch.Tensor,
    B: torch.Tensor,
    beta: float,
    gamma: float,
    synergy: float = 1.0,
) -> torch.Tensor:
    """Advance one discrete SIR step on a hypergraph.

    For each hyperedge e, let m_e be the number of infected members.  For a
    susceptible node i in e, the infection pressure contributed by e is

        (m_e / (|e|-1)) ** synergy.

    Pressures from all incident hyperedges are added, and converted to an
    infection probability using

        p_i = 1 - exp(-beta * pressure_i).

    An infected node recovers independently with probability gamma.
    Recovered nodes remain recovered.

    With synergy > 1, simultaneous infection of multiple members of the same
    hyperedge has a stronger higher-order effect.  synergy=1 gives a simpler
    additive group exposure.
    """
    if x.ndim != 1 or x.numel() != B.shape[0]:
        raise ValueError(f"x must have shape [{B.shape[0]}], got {tuple(x.shape)}")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0,1]")
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0,1]")
    if synergy <= 0:
        raise ValueError("synergy must be > 0")

    infected = (x == I_STATE).to(B.dtype)  # [N]
    infected_per_edge = B.transpose(0, 1) @ infected  # [E]
    edge_size = B.sum(dim=0).clamp_min(1.0)  # [E]

    # For a susceptible target, its own state is not infected, so m_e already
    # counts only infected co-members.  Divide by k-1 to normalize group exposure.
    co_member_count = (edge_size - 1.0).clamp_min(1.0)
    edge_fraction = (infected_per_edge / co_member_count).clamp(0.0, 1.0)
    edge_pressure = edge_fraction.pow(float(synergy))  # [E]
    node_pressure = B @ edge_pressure  # [N]
    p_infect = 1.0 - torch.exp(-float(beta) * node_pressure)
    p_infect = p_infect.clamp(0.0, 1.0)

    y = x.clone()
    susceptible = x == S_STATE
    infected_mask = x == I_STATE

    new_infected = susceptible & (torch.rand(x.shape, device=x.device) < p_infect)
    new_recovered = infected_mask & (
        torch.rand(x.shape, device=x.device) < float(gamma)
    )

    y[new_infected] = I_STATE
    y[new_recovered] = R_STATE
    # R stays R automatically.
    return y


@torch.no_grad()
def generate_sir_pairs(
    B: torch.Tensor,
    num_pairs: int,
    episode_length: int,
    beta: float,
    gamma: float,
    synergy: float,
    init_infected_fraction: float,
    init_recovered_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate informative one-step SIR pairs with periodic epidemic resets.

    A single long SIR trajectory quickly reaches the absorbing no-infection state.
    For AIDD structure learning that would create thousands of nearly useless
    S->S / R->R samples.  We therefore generate short SIR episodes and reset the
    initial outbreak between episodes.  Every saved pair still satisfies exactly

        Y[r] = one_SIR_step(X[r], B).

    Rows across episode boundaries are not required to be consecutive in time;
    the AIDD training loader treats each row as a one-step supervised pair.
    """
    if num_pairs <= 0:
        raise ValueError("num_pairs must be > 0")
    if episode_length <= 0:
        raise ValueError("episode_length must be > 0")

    N = B.shape[0]
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []

    while len(xs) < num_pairs:
        x = initialize_sir_state(
            N=N,
            infected_fraction=init_infected_fraction,
            recovered_fraction=init_recovered_fraction,
            device=str(B.device),
        )

        for _ in range(episode_length):
            if len(xs) >= num_pairs:
                break

            # If infection has already died out, restart instead of adding a long
            # sequence of trivial absorbing-state pairs.
            if not torch.any(x == I_STATE):
                break

            y = hypergraph_sir_step(
                x=x,
                B=B,
                beta=beta,
                gamma=gamma,
                synergy=synergy,
            )
            xs.append(x.clone())
            ys.append(y.clone())
            x = y

    X = torch.stack(xs, dim=0).to(torch.long)
    Y = torch.stack(ys, dim=0).to(torch.long)
    return X, Y


def validate_sir_pairs(X: torch.Tensor, Y: torch.Tensor) -> None:
    """Validate shape, labels, and legal SIR transitions."""
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
        raise ValueError(f"X/Y must have equal [samples,N] shapes, got {X.shape}, {Y.shape}")

    valid_states = torch.tensor([S_STATE, I_STATE, R_STATE], device=X.device)
    if not torch.all(torch.isin(X, valid_states)):
        raise ValueError("X contains a state outside {0,1,2}")
    if not torch.all(torch.isin(Y, valid_states)):
        raise ValueError("Y contains a state outside {0,1,2}")

    # Legal SIR transitions: S->{S,I}, I->{I,R}, R->R.
    invalid = (
        ((X == S_STATE) & ~((Y == S_STATE) | (Y == I_STATE)))
        | ((X == I_STATE) & ~((Y == I_STATE) | (Y == R_STATE)))
        | ((X == R_STATE) & (Y != R_STATE))
    )
    if torch.any(invalid):
        count = int(invalid.sum().item())
        raise ValueError(f"Found {count} illegal SIR node transitions")


def save_state_matrix(data: torch.Tensor, path: str | Path) -> Path:
    """Save discrete [samples,N] integer states as whitespace-separated text."""
    if data.ndim != 2:
        raise ValueError(f"data must be [samples,N], got {tuple(data.shape)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = data.detach().cpu().to(torch.int64).tolist()
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(" ".join(map(str, row)) + "\n")
    return path


def generate_and_save_dataset(
    output_dir: str | Path,
    prefix: str,
    N: int,
    E: int,
    hyperedge_size: int,
    num_pairs: int,
    episode_length: int,
    beta: float,
    gamma: float,
    synergy: float,
    init_infected_fraction: float,
    init_recovered_fraction: float,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[Path, Path, Path]:
    """Generate hypergraph topology plus SIR X/Y and save three model files."""
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    B = generate_fixed_size_incidence_matrix(
        N=N,
        E=E,
        hyperedge_size=hyperedge_size,
        device=device,
    )

    X, Y = generate_sir_pairs(
        B=B,
        num_pairs=num_pairs,
        episode_length=episode_length,
        beta=beta,
        gamma=gamma,
        synergy=synergy,
        init_infected_fraction=init_infected_fraction,
        init_recovered_fraction=init_recovered_fraction,
    )
    validate_sir_pairs(X, Y)

    structure_path = output_dir / f"{prefix}.txt"
    x_path = output_dir / f"{prefix}_SIR_x.txt"
    y_path = output_dir / f"{prefix}_SIR_y.txt"

    save_hypergraph_structure(B, structure_path)
    save_state_matrix(X, x_path)
    save_state_matrix(Y, y_path)

    unique_x = sorted(int(v) for v in torch.unique(X).cpu().tolist())
    unique_y = sorted(int(v) for v in torch.unique(Y).cpu().tolist())
    edge_sizes = B.sum(dim=0)

    print("==============================")
    print("Hypergraph SIR generation complete")
    print("==============================")
    print(f"dataset dir : {output_dir}")
    print(f"B shape     : {tuple(B.shape)}  [N,E], not written directly")
    print(
        f"hyperedges  : E={E}, size min/mean/max="
        f"{int(edge_sizes.min().item())}/{edge_sizes.mean().item():.2f}/{int(edge_sizes.max().item())}"
    )
    print(f"X shape     : {tuple(X.shape)}, states={unique_x} -> {x_path}")
    print(f"Y shape     : {tuple(Y.shape)}, states={unique_y} -> {y_path}")
    print(f"structure   : {structure_path}")
    print("SIR transition validation: PASS")
    print("Each row pair satisfies Y[r] = one hypergraph-SIR step from X[r].")

    return structure_path, x_path, y_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a k-uniform hypergraph and discrete SIR X/Y data for AIDD."
    )
    # Folder name and prefix intentionally match by default, which fits
    # dynamics_predict.py's network_name = data_dir.name convention.
    parser.add_argument("--output-dir", type=str, default="generated_data/hypergraph_N200")
    parser.add_argument("--prefix", type=str, default="hypergraph_N200")
    parser.add_argument("--N", type=int, default=200, help="number of nodes")
    parser.add_argument("--E", type=int, default=400, help="number of hyperedges")
    parser.add_argument("--hyperedge-size", type=int, default=3, help="nodes per hyperedge")
    parser.add_argument("--num-pairs", type=int, default=5000, help="number of one-step X/Y pairs")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=12,
        help="maximum SIR steps before resetting to a new outbreak",
    )
    parser.add_argument("--beta", type=float, default=0.20, help="hypergraph infection strength")
    parser.add_argument("--gamma", type=float, default=0.10, help="I->R recovery probability per step")
    parser.add_argument(
        "--synergy",
        type=float,
        default=2.0,
        help="higher-order infection exponent; >1 strengthens multi-infected hyperedge effects",
    )
    parser.add_argument("--init-infected-fraction", type=float, default=0.10)
    parser.add_argument("--init-recovered-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    output_dir = Path(args.output_dir)
    # This does not forbid custom names, but warns because the graph loader shown by
    # the user derives network_name from the directory basename.
    if output_dir.name != args.prefix:
        print(
            f"[warning] output directory basename '{output_dir.name}' != prefix '{args.prefix}'. "
            "For CoND/AIDD-style loading, it is simplest to keep them identical."
        )

    generate_and_save_dataset(
        output_dir=output_dir,
        prefix=args.prefix,
        N=args.N,
        E=args.E,
        hyperedge_size=args.hyperedge_size,
        num_pairs=args.num_pairs,
        episode_length=args.episode_length,
        beta=args.beta,
        gamma=args.gamma,
        synergy=args.synergy,
        init_infected_fraction=args.init_infected_fraction,
        init_recovered_fraction=args.init_recovered_fraction,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
