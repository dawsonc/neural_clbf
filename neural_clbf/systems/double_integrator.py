"""Define a dymamical system for an inverted pendulum"""
from typing import Tuple, Optional, List

import torch

from .control_affine_system import ControlAffineSystem
from neural_clbf.systems.utils import grav, Scenario, ScenarioList


class DoubleIntegrator1D(ControlAffineSystem):
    """
    Represents a 1d double integrator.

    The system has state

        x = [x, x_dot]

    representing the position and velocity of the mass, and it
    has control inputs

        u = [a]

    representing the acceleration applied.   
    """

    # Number of states and controls
    N_DIMS = 2
    N_CONTROLS = 1

    # State indices
    X = 0
    X_DOT = 1
    # Control indices
    U = 0

    def __init__(
        self,
        nominal_params: Scenario,
        dt: float = 0.05,
        controller_dt: Optional[float] = None,
        scenarios: Optional[ScenarioList] = None,
    ):
        """
        Initialize the double integrator.

        args:
            nominal_params: a dictionary giving the parameter values for the system.
                            Expected to be empty.
            dt: the timestep to use for the simulation
            controller_dt: the timestep for the LQR discretization. Defaults to dt
        raises:
            ValueError if nominal_params are not valid for this system
        """
        # Precompute the matrices A and B for the dynamics.
        self._matrix_A = torch.tensor([[1, dt], [0, 1]])
        self._matrix_B = torch.tensor([[dt ** 2 / 2], [dt]])

        super().__init__(
            nominal_params, dt=dt, controller_dt=controller_dt, scenarios=scenarios
        )



    def validate_params(self, params: Scenario) -> bool:
        """Dummy check  that no parameters are passed.

        args:
            params: a dictionary giving the parameter values for the system.
                    It is expected to be empty.
        returns:
            True if parameters dict is empty, False otherwise.
        """
        return isinstance(params, dict) and len(params) == 0

    @property
    def n_dims(self) -> int:
        return DoubleIntegrator1D.N_DIMS

    @property
    def angle_dims(self) -> List[int]:
        return []

    @property
    def n_controls(self) -> int:
        return DoubleIntegrator1D.N_CONTROLS

    @property
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_dims)
        upper_limit[DoubleIntegrator1D.X] = 1.5
        upper_limit[DoubleIntegrator1D.X_DOT] = 2.0

        lower_limit = -upper_limit

        return (upper_limit, lower_limit)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        # define upper and lower limits based around the nominal equilibrium input
        upper_limit = torch.ones(self.n_controls)
        lower_limit = -upper_limit

        return (upper_limit, lower_limit)

    def safe_mask(self, x):
        """Return the mask of x indicating safe regions for the obstacle task.

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        return torch.abs(x[:, DoubleIntegrator1D.X]) <= 1.0

    def unsafe_mask(self, x):
        """Return the mask of x indicating unsafe regions for the obstacle task

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        return ~self.safe_mask(x)

    def goal_mask(self, x):
        """Return the mask of x indicating points in the goal set

        args:
            x: a tensor of (batch_size, self.n_dims) points in the state space
        returns:
            a tensor of (batch_size,) booleans indicating whether the corresponding
            point is in this region.
        """
        return x.norm(dim=-1) <= 0.5

    def _f(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            f: bs x self.n_dims x 1 tensor
        """
        return (x @ self._matrix_A.T).unsqueeze(-1)

    def _g(self, x: torch.Tensor, params: Scenario):
        """
        Return the control-independent part of the control-affine dynamics.

        args:
            x: bs x self.n_dims tensor of state
            params: a dictionary giving the parameter values for the system. If None,
                    default to the nominal parameters used at initialization
        returns:
            g: bs x self.n_dims x self.n_controls tensor
        """        
        return self._matrix_B.repeat(x.shape[0], 1, 1)
