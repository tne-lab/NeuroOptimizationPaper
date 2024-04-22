import warnings
from typing import Optional, Dict, Union, List, Tuple

import torch
from botorch import settings
from botorch.acquisition import AcquisitionFunction
from botorch.exceptions import SamplingWarning, BotorchWarning, BadInitialCandidatesWarning
from botorch.optim.initializers import is_nonnegative, initialize_q_batch_nonneg, initialize_q_batch, \
    sample_points_around_best
from botorch.optim.utils import fix_features
from botorch.utils.sampling import get_polytope_samples
from torch import Tensor
from torch.quasirandom import SobolEngine


def gen_batch_initial_conditions_nonlinear(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features: Optional[Dict[int, float]] = None,
    options: Optional[Dict[str, Union[bool, float, int]]] = None,
    inequality_constraints: Optional[
        List[Tuple[Tensor, Tensor, float]]
    ] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    nonlinear_constraint: Optional[callable] = None,
) -> Tensor:
    r"""Generate a batch of initial conditions for random-restart optimziation.

    TODO: Support t-batches of initial conditions.

    Args:
        acq_function: The acquisition function to be optimized.
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates to consider.
        num_restarts: The number of starting points for multistart acquisition
            function optimization.
        raw_samples: The number of raw samples to consider in the initialization
            heuristic. Note: if `sample_around_best` is True (the default is False),
            then `2 * raw_samples` samples are used.
        fixed_features: A map `{feature_index: value}` for features that
            should be fixed to a particular value during generation.
        options: Options for initial condition generation. For valid options see
            `initialize_q_batch` and `initialize_q_batch_nonneg`. If `options`
            contains a `nonnegative=True` entry, then `acq_function` is
            assumed to be non-negative (useful when using custom acquisition
            functions). In addition, an "init_batch_limit" option can be passed
            to specify the batch limit for the initialization. This is useful
            for avoiding memory limits when computing the batch posterior over
            raw samples.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`.
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`.

    Returns:
        A `num_restarts x q x d` tensor of initial conditions.

    Example:
        >>> qEI = qExpectedImprovement(model, best_f=0.2)
        >>> bounds = torch.tensor([[0.], [1.]])
        >>> Xinit = gen_batch_initial_conditions(
        >>>     qEI, bounds, q=3, num_restarts=25, raw_samples=500
        >>> )
    """
    if bounds.isinf().any():
        raise NotImplementedError(
            "Currently only finite values in `bounds` are supported "
            "for generating initial conditions for optimization."
        )
    options = options or {}
    sample_around_best = options.get("sample_around_best", False)
    if sample_around_best and equality_constraints:
        raise UnsupportedError(
            "Option 'sample_around_best' is not supported when equality"
            "constraints are present."
        )
    seed: Optional[int] = options.get("seed")
    batch_limit: Optional[int] = options.get(
        "init_batch_limit", options.get("batch_limit")
    )
    factor, max_factor = 1, 5
    init_kwargs = {}
    device = bounds.device
    if "eta" in options:
        init_kwargs["eta"] = options.get("eta")
    if options.get("nonnegative") or is_nonnegative(acq_function):
        init_func = initialize_q_batch_nonneg
        if "alpha" in options:
            init_kwargs["alpha"] = options.get("alpha")
    else:
        init_func = initialize_q_batch

    q = 1 if q is None else q
    # the dimension the samples are drawn from
    effective_dim = bounds.shape[-1] * q
    if effective_dim > SobolEngine.MAXDIM and settings.debug.on():
        warnings.warn(
            f"Sample dimension q*d={effective_dim} exceeding Sobol max dimension "
            f"({SobolEngine.MAXDIM}). Using iid samples instead.",
            SamplingWarning,
        )

    while factor < max_factor:
        with warnings.catch_warnings(record=True) as ws:
            n = raw_samples * factor

            if sample_around_best:
                BotorchWarning(
                    "sample_around_best is True but there are "
                    "non-linear constraints provided. This might lead to "
                    "unexpected behavior since sampling points around "
                    "best has some random behavior which could violate "
                    "the non-linear constraints"
                )

            need = num_restarts * q
            all_points = torch.tensor([]).cpu()
            ndims = bounds.shape[1]
            counter = 0
            while all_points.shape[0] <= need:
                X_rnd = (
                    get_polytope_samples(
                        n=n * q,
                        bounds=bounds,
                        inequality_constraints=inequality_constraints,
                        equality_constraints=equality_constraints,
                        seed=seed,
                        n_burnin=options.get("n_burnin", 10000),
                        thinning=options.get("thinning", 32),
                    )
                    .view(n, q, -1)
                    .cpu()
                )
                X_rnd = X_rnd.reshape(-1, ndims)
                where = torch.where(nonlinear_constraint(X_rnd) >= 0)[0]
                all_points = torch.cat([all_points, X_rnd[where, :]], axis=0)
                counter += 1
                if seed is not None:
                    seed += 1
                if counter >= 1000:
                    BotorchWarning(
                        "1000 iterations of trying to satisfy the "
                        "provided non-linear constraint. Recommend "
                        "revisiting the parameters of this constraint, "
                        "as it appears hard to satisfy it."
                    )
            X_rnd = all_points[:need, :].reshape(num_restarts, q, ndims)

            # sample points around best
            if sample_around_best:
                X_best_rnd = sample_points_around_best(
                    acq_function=acq_function,
                    n_discrete_points=n * q,
                    sigma=options.get("sample_around_best_sigma", 1e-3),
                    bounds=bounds,
                    subset_sigma=options.get(
                        "sample_around_best_subset_sigma", 1e-1
                    ),
                    prob_perturb=options.get(
                        "sample_around_best_prob_perturb"
                    ),
                )
                if X_best_rnd is not None:
                    X_rnd = torch.cat(
                        [
                            X_rnd,
                            X_best_rnd.view(n, q, bounds.shape[-1]).cpu(),
                        ],
                        dim=0,
                    )
            X_rnd = fix_features(X_rnd, fixed_features=fixed_features)
            with torch.no_grad():
                if batch_limit is None:
                    batch_limit = X_rnd.shape[0]
                Y_rnd_list = []
                start_idx = 0
                while start_idx < X_rnd.shape[0]:
                    end_idx = min(start_idx + batch_limit, X_rnd.shape[0])
                    Y_rnd_curr = acq_function(
                        X_rnd[start_idx:end_idx].to(device=device)
                    ).cpu()
                    Y_rnd_list.append(Y_rnd_curr)
                    start_idx += batch_limit
                Y_rnd = torch.cat(Y_rnd_list)
            batch_initial_conditions = init_func(
                X=X_rnd, Y=Y_rnd, n=num_restarts, **init_kwargs
            ).to(device=device)
            if not any(
                issubclass(w.category, BadInitialCandidatesWarning) for w in ws
            ):
                return batch_initial_conditions
            if factor < max_factor:
                factor += 1
                if seed is not None:
                    seed += 1  # make sure to sample different X_rnd
    warnings.warn(
        "Unable to find non-zero acquisition function values - initial conditions "
        "are being selected randomly.",
        BadInitialCandidatesWarning,
    )
    return batch_initial_conditions
