"""Five elemental Petri net subnets from §5 of the architecture spec.

Continuous relaxation of the token firing rule (spec §4.2):

    activation(t) = sigmoid( sum_p w(p,t) * a(p) - theta(t) )
    a(p)          = sum_{t: (t,p) in F} activation(t) * w(t,p)

Each place's activation a(p) is a scalar in [0,1] (spec §4.1) carried as
a tensor of shape (batch,) so that a forward pass can score many process
instances at once.

Forward passes return a dict keyed by Petri-net element name (the same
names that appear in the diagrams in §5), so a caller can inspect any
transition activation or place activation without having to remember a
positional convention.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _fire(weighted_input: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(weighted_input - threshold)


class SequentialSubnet(nn.Module):
    """Subnet 1 — sequential execution.

        [P_before] --T_step--> [P_after]
    """

    def __init__(self) -> None:
        super().__init__()
        self.w_in = nn.Parameter(torch.tensor(1.0))
        self.theta_step = nn.Parameter(torch.tensor(0.0))
        self.w_out = nn.Parameter(torch.tensor(1.0))

    def forward(self, p_before: torch.Tensor) -> dict[str, torch.Tensor]:
        t_step = _fire(self.w_in * p_before, self.theta_step)
        p_after = t_step * self.w_out
        return {"T_step": t_step, "P_after": p_after}


class XORSubnet(nn.Module):
    """Subnet 2 — exclusive choice.

                       --T_route_A--> [P_path_A]
        [P_decision] <
                       --T_route_B--> [P_path_B]

    Both transitions share P_decision as their input place. Soft routing
    is what the continuous relaxation produces; the structural constraint
    (only these two declared paths exist) is enforced by construction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w_A_in = nn.Parameter(torch.tensor(1.0))
        self.theta_A = nn.Parameter(torch.tensor(0.0))
        self.w_A_out = nn.Parameter(torch.tensor(1.0))

        self.w_B_in = nn.Parameter(torch.tensor(-1.0))
        self.theta_B = nn.Parameter(torch.tensor(0.0))
        self.w_B_out = nn.Parameter(torch.tensor(1.0))

    def forward(self, p_decision: torch.Tensor) -> dict[str, torch.Tensor]:
        t_A = _fire(self.w_A_in * p_decision, self.theta_A)
        t_B = _fire(self.w_B_in * p_decision, self.theta_B)
        p_path_A = t_A * self.w_A_out
        p_path_B = t_B * self.w_B_out
        return {
            "T_route_A": t_A,
            "T_route_B": t_B,
            "P_path_A": p_path_A,
            "P_path_B": p_path_B,
        }


class AndSplitSubnet(nn.Module):
    """Subnet 3 — parallel split (AND-gateway).

                      --> [P_branch_A]
        [P_ready] -- T_spawn
                      --> [P_branch_B]

    A single transition with two output arcs; both branches activate
    simultaneously when T_spawn fires.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w_in = nn.Parameter(torch.tensor(1.0))
        self.theta_spawn = nn.Parameter(torch.tensor(0.0))
        self.w_out_A = nn.Parameter(torch.tensor(1.0))
        self.w_out_B = nn.Parameter(torch.tensor(1.0))

    def forward(self, p_ready: torch.Tensor) -> dict[str, torch.Tensor]:
        t_spawn = _fire(self.w_in * p_ready, self.theta_spawn)
        return {
            "T_spawn": t_spawn,
            "P_branch_A": t_spawn * self.w_out_A,
            "P_branch_B": t_spawn * self.w_out_B,
        }


class AndJoinSubnet(nn.Module):
    """Subnet 4 — synchronisation (AND-join).

        [P_branch_A] --\\
                         T_merge --> [P_unified]
        [P_branch_B] --/

    Spec §5 calls this the hardest subnet to relax: a partial firing
    (one branch complete, the other not) must not propagate. We bias
    initialisation toward a high threshold and use a sharpness parameter
    on the sigmoid so the activation function approaches a step.
    """

    def __init__(self, sharpness: float = 4.0) -> None:
        super().__init__()
        self.w_A_in = nn.Parameter(torch.tensor(1.0))
        self.w_B_in = nn.Parameter(torch.tensor(1.0))
        self.theta_merge = nn.Parameter(torch.tensor(1.5))
        self.w_out = nn.Parameter(torch.tensor(1.0))
        self.sharpness = sharpness

    def forward(
        self, p_branch_A: torch.Tensor, p_branch_B: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        weighted = self.w_A_in * p_branch_A + self.w_B_in * p_branch_B
        t_merge = _fire(self.sharpness * weighted, self.sharpness * self.theta_merge)
        return {"T_merge": t_merge, "P_unified": t_merge * self.w_out}


class SagaSubnet(nn.Module):
    """Subnet 5 — saga compensation.

        [P_active] --T_succeed--> [P_complete]
        [P_active] --T_fail-----> [P_compensating]
        [P_compensating] --T_compensate--> [P_initial]

    Two competing transitions out of P_active (success / failure), and a
    compensation transition closing the loop back to P_initial. The whole
    subnet is a feed-forward DAG when unrolled for one execution step.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w_succeed_in = nn.Parameter(torch.tensor(1.0))
        self.theta_succeed = nn.Parameter(torch.tensor(0.0))
        self.w_succeed_out = nn.Parameter(torch.tensor(1.0))

        self.w_fail_in = nn.Parameter(torch.tensor(-1.0))
        self.theta_fail = nn.Parameter(torch.tensor(0.0))
        self.w_fail_out = nn.Parameter(torch.tensor(1.0))

        self.w_compensate_in = nn.Parameter(torch.tensor(1.0))
        self.theta_compensate = nn.Parameter(torch.tensor(0.0))
        self.w_compensate_out = nn.Parameter(torch.tensor(1.0))

    def forward(self, p_active: torch.Tensor) -> dict[str, torch.Tensor]:
        t_succeed = _fire(self.w_succeed_in * p_active, self.theta_succeed)
        t_fail = _fire(self.w_fail_in * p_active, self.theta_fail)

        p_complete = t_succeed * self.w_succeed_out
        p_compensating = t_fail * self.w_fail_out

        t_compensate = _fire(
            self.w_compensate_in * p_compensating, self.theta_compensate
        )
        p_initial = t_compensate * self.w_compensate_out

        return {
            "T_succeed": t_succeed,
            "T_fail": t_fail,
            "T_compensate": t_compensate,
            "P_complete": p_complete,
            "P_compensating": p_compensating,
            "P_initial": p_initial,
        }
