from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

from neural_clbf.controllers import NeuralCLBFController
from neural_clbf.controllers.neural_cbf_controller import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import DoubleIntegrator1D
from neural_clbf.experiments import (
    ExperimentSuite,
    CLFContourExperiment,
    RolloutStateSpaceExperiment,
)
from neural_clbf.systems.double_integrator import DoubleIntegrator1D
from neural_clbf.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

batch_size = 64
controller_period = 0.05

start_x = torch.tensor(
    [
        [0.5, 0.5],
        [-0.2, 1.0],
        [0.2, -1.0],
        [-0.2, -1.0],
    ]
)
simulation_dt = 0.05


def main(args):
    nominal_params = {}
    scenarios = [nominal_params]

    # Define the dynamics model    
    dynamics_model = DoubleIntegrator1D(
        nominal_params=nominal_params,
        dt=simulation_dt,
        controller_dt=controller_period,        
        scenarios=scenarios,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-1.0, 1.0),  # x
        (-1.0, 1.0),  # x_dot
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=0,
        trajectory_length=1,
        fixed_samples=10000,
        max_points=100000,
        val_split=0.1,
        batch_size=64,
    )

    # Define the experiment suite
    V_contour_experiment = CLFContourExperiment(
        "V_Contour",
        domain=[(-1.5, 1.5), (-2.0, 2.0)],
        n_grid=30,
        x_axis_index=DoubleIntegrator1D.X,
        y_axis_index=DoubleIntegrator1D.X_DOT,
        x_axis_label="$x$",
        y_axis_label="$\\dot{x}$",
        plot_unsafe_region=True,        
    )
    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        DoubleIntegrator1D.X,
        "$x$",
        DoubleIntegrator1D.X_DOT,
        "$\\dot{x}$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite([V_contour_experiment, rollout_experiment])

    # Initialize the controller
    cbf_controller = NeuralCBFController(
        dynamics_model=dynamics_model,        
        scenarios=scenarios,
        datamodule=data_module,
        experiment_suite=experiment_suite,
        cbf_hidden_layers=2,
        cbf_hidden_size=64,
        cbf_lambda=1.0,
        cbf_relaxation_penalty=50.0,        
        controller_period=controller_period,
        primal_learning_rate=1e-3,
        scale_parameter=10.0,
        learn_shape_epochs=0,
        use_relu=False,
        disable_gurobi=True,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/double_integrator",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        max_epochs=51,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(cbf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
