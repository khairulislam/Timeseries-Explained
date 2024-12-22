import copy
import torch as th

from captum.attr._utils.attribution import PerturbationAttribution
from captum.log import log_usage
from captum._utils.common import (
    _format_baseline,
    _format_inputs,
    _format_output,
    _is_tuple,
    _validate_input,
)
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from typing import Any, Callable, Tuple

from tint.utils import TensorDataset, _add_temporal_mask, default_collate
from attrs.gatemasknn import GateMaskNet


class GateMask(PerturbationAttribution):
    """
    GateMask.

    ContraLSP, a locally sparse model that introduces counterfactual 
    samples to build uninformative perturbations but keeps distribution 
    using contrastive learning. Furthermore, we incorporate sample-specific 
    sparse gates to generate more binary-skewed and smooth
    masks, which easily integrate temporal trends and select the salient 
    features parsimoniously.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.

    References:
        #. `Explaining Time Series via Contrastive and Locally Sparse Perturbations (ContraLSP) <https://arxiv.org/abs/2305.18840>`
    """

    def __init__(self, forward_func: Callable) -> None:
        super().__init__(forward_func=forward_func)

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        trainer: Trainer = None,
        mask_net: GateMaskNet = None,
        batch_size: int = 32,
        temporal_additional_forward_args: Tuple[bool] = None,
        return_temporal_attributions: bool = False,
        win_size: int = 5,
        sigma: float = 0.5,
    ) -> TensorOrTupleOfTensorsGeneric:

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs = _format_inputs(inputs)

        # Format and validate baselines
        baselines = _format_baseline(baselines, inputs)
        _validate_input(inputs, baselines)

        # Init trainer if not provided
        if trainer is None:
            trainer = Trainer(max_epochs=100)
        else:
            trainer = copy.deepcopy(trainer)

        # Assert only one input, as the Retain only accepts one
        assert (
            len(inputs) == 1
        ), "Multiple inputs are not accepted for this method"
        data = inputs[0]
        baseline = baselines[0]

        # If return temporal attr, we expand the input data
        # and multiply it with a lower triangular mask
        if return_temporal_attributions:
            data, additional_forward_args, _ = _add_temporal_mask(
                inputs=data,
                additional_forward_args=additional_forward_args,
                temporal_additional_forward_args=temporal_additional_forward_args,
            )

        # Init MaskNet if not provided
        if mask_net is None:
            mask_net = GateMaskNet(
                forward_func=self.forward_func
            )

        # Init model
        mask_net.net.init(input_size=data.shape, batch_size=batch_size, win_size=win_size,
                          sigma=sigma, n_epochs=trainer.max_epochs)

        # Prepare data
        dataloader = DataLoader(
            TensorDataset(
                *(data, data, baseline, target, *additional_forward_args)
                if additional_forward_args is not None
                else (data, data, baseline, target, None)
            ),
            batch_size=batch_size,
            collate_fn=default_collate,
        )

        # Fit model
        trainer.fit(mask_net, train_dataloaders=dataloader)

        # Set model to eval mode and cast it to device
        mask_net.eval()
        mask_net.to(data.device)

        # Get attributions as mask representation
        attributions = mask_net.net.representation(data)
        # self.learn_sig = mask_net.net.refactor_mask(mask_net.net.mask, data)
        # self.no_sig = mask_net.net.mask+0.5

        # Reshape representation if temporal attributions
        if return_temporal_attributions:
            attributions = attributions.reshape(
                (-1, data.shape[1]) + data.shape[1:]
            )

        # Reshape as a tuple
        attributions = (attributions,)

        # Format attributions and return
        return _format_output(is_inputs_tuple, attributions)
