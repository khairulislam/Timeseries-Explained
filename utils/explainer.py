import torch
import numpy as np
from utils.tools import reshape_over_output_horizon, round_up
from pytorch_lightning import Trainer
from exp.exp_basic import dual_input_users
from attrs.gatemasknn import GateMaskNet
from tint.models import MLP, RNN
from pytorch_lightning.callbacks import EarlyStopping
from tint.attr.models import JointFeatureGeneratorNet
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

def get_total_data(dataloader, device, add_x_mark=True):        
    if add_x_mark:
        return (
            torch.vstack([item[0] for item in dataloader]).float().to(device), 
            torch.vstack([item[2] for item in dataloader]).float().to(device)
        )
    else:
        return torch.vstack([item[0] for item in dataloader]).float().to(device)

def get_baseline(inputs, mode='random'):
    if type(inputs) == tuple:
        return tuple([get_baseline(input, mode) for input in inputs])
    
    batch_size, seq_len, n_features = inputs.shape[0], inputs.shape[1], inputs.shape[2]    
    device = inputs.device
    
    if mode =='zero': baselines = torch.zeros_like(inputs, device=device).float()
    elif mode == 'random': 
        baselines = torch.randn_like(inputs, device=device).float()
        # baselines = torch.normal(0, 1.2, size=inputs.shape, device=device).float()
    elif mode == 'aug':
        inputs = inputs.reshape((-1, n_features))
        baselines = torch.zeros_like(inputs, device=device).float()
        
        for f in range(n_features):
            choices = inputs[:, f]
            sampled_index = np.random.choice(
                range(inputs.shape[0]), size=inputs.shape[0], replace=True
            )
            baselines[:, f] = choices[sampled_index]
        baselines = baselines.reshape((batch_size, seq_len, n_features))
    elif mode == 'normal':
        means = torch.mean(inputs, dim=(0, 1), keepdim=True).repeat(batch_size, seq_len, 1)
        std = torch.std(inputs, dim=(0, 1), keepdim=True).repeat(batch_size, seq_len, 1)
        
        baselines = torch.normal(means, std).float()
        # .repeat(
        #     batch_size, inputs.shape[1], 1
        # ).float()
    elif mode == 'mean': 
        baselines = torch.mean(
                inputs, axis=0
        ).repeat(batch_size, 1, 1).float()
        
    elif mode == 'gen':
        trainer = Trainer(
            logger=False, enable_checkpointing=False,
            enable_progress_bar=False, max_epochs=100,
            enable_model_summary=False,accelerator='auto',
            callbacks=[EarlyStopping('train_loss', patience=5, mode='min', verbose=False)],
            # precision='64'
        )
        generator = JointFeatureGeneratorNet()
        n_features = inputs.shape[-1]
        generator.net.init(feature_size=n_features)
        generator.net.to(device)
        generator.to(device)
        
        dataloader = DataLoader(
            TensorDataset(inputs),
            batch_size=batch_size,
        )
        # Train generator
        trainer.fit(
            generator, train_dataloaders=dataloader
        )
        generator.eval()
        generator.net.to(device)
        
        baselines = torch.randn_like(inputs, device=device).float()
        for t in range(1, inputs.shape[1]):
            baselines[:, t] = generator.net.forward(inputs[:, :t])
    else:
        print(f'baseline mode options: [zero, random, aug, mean, normal]')
        raise NotImplementedError
    
    return baselines

def compute_attr_with_trainer(
    inputs, explainer,
    additional_forward_args, targets
): # the parameters ensure Trainer doesn't flood the output with logs and create log folders
    trainer = Trainer(
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, max_epochs=100,accelerator="gpu",
        enable_model_summary=False
    )
    if type(inputs) == tuple:
        new_additional_forward_args = tuple([
            input for input in inputs[1:]
        ]) + additional_forward_args
        
        # output is a tuple of length 1, since only one input is used
        attr = explainer.attribute(
            inputs=inputs[0],
            additional_forward_args=new_additional_forward_args,
            trainer=trainer
        )
        # output isn't for each target, unlike other methods
        # batch_size x seq_len x features -> (targets x batch_size) x seq_len x features
        attr = attr.repeat(targets, 1, 1)
        
        attr = tuple([attr] + [
            torch.zeros(
                (inputs[i].shape[0]*targets, inputs[i].shape[1], inputs[i].shape[2]), 
                device=inputs[i].device) for i in range(1, len(inputs))]
        )
    else: 
        # batch_size x seq_len x features
        attr = explainer.attribute(
            inputs=inputs,
            additional_forward_args=additional_forward_args,
            trainer=trainer
        )
        # output isn't for each target, unlike other methods
        # batch_size x seq_len x features -> (targets x batch_size) x seq_len x features
        attr = attr.repeat(targets, 1, 1)
        
    return attr

def compute_attr_with_gatemask(
    inputs, baselines, explainer,
    additional_forward_args, args, targets
): 
    trainer = Trainer(
        logger=False, enable_checkpointing=False,
        enable_progress_bar=False, max_epochs=50,
        enable_model_summary=False,accelerator="auto",
        callbacks=[EarlyStopping('train_loss', patience=10, mode='min', verbose=False)],
    )
    
    mask = GateMaskNet(
        forward_func=explainer.forward_func,
        model=torch.nn.Sequential(
            RNN(
                input_size=args.n_features,
                rnn="gru",
                hidden_size=args.n_features,
                bidirectional=True,
            ),
            MLP([2 * args.n_features, args.n_features]),
        ),
        lambda_1=1,  # 0.1 for our lambda is suitable
        lambda_2=1,
        optim="adam",
        lr=0.01,
    )
    
    
    if type(inputs) == tuple:
        new_additional_forward_args = tuple([
            input for input in inputs[1:]
        ]) + additional_forward_args
        mask.to(inputs[0].device)
        
        # output is a tuple of length 1, since only one input is used
        attr = explainer.attribute(
            inputs=inputs[0],
            baselines=baselines[0],
            additional_forward_args=new_additional_forward_args,
            mask_net=mask, 
            batch_size=args.batch_size,
            trainer=trainer
        )
        # output isn't for each target, unlike other methods
        # batch_size x seq_len x features -> (targets x batch_size) x seq_len x features
        attr = attr.repeat(targets, 1, 1)
        
        attr = tuple([attr] + [
            torch.zeros(
                (inputs[i].shape[0]*targets, inputs[i].shape[1], inputs[i].shape[2]), 
                device=inputs[i].device) for i in range(1, len(inputs))]
        )
    else: 
        mask.to(inputs.device)
        # batch_size x seq_len x features
        attr = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            mask_net=mask, 
            batch_size=args.batch_size,
            trainer=trainer
        )
        # output isn't for each target, unlike other methods
        # batch_size x seq_len x features -> (targets x batch_size) x seq_len x features
        attr = attr.repeat(targets, 1, 1)
        
    return attr

def compute_attr_with_gradient(
    name, inputs, baselines, explainer,
    additional_forward_args, args, targets
): 
    attr_list = []
        
    for target in range(targets):
        if name in ['deep_lift', 'integrated_gradients', 'gradient_shap']:
            
            # these models use the multiple inputs in the forward function
            if type(inputs) == tuple and args.model not in dual_input_users:
                new_additional_forward_args = tuple([
                    input for input in inputs[1:]
                ]) + additional_forward_args
                
                # output is a tuple of length 1, since only one input is used
                attr = explainer.attribute(
                    inputs=inputs[0], baselines=baselines[0], target=target,
                    additional_forward_args=new_additional_forward_args
                )
                
                attr = tuple([attr] + [
                    torch.zeros_like(inputs[i], device=inputs[i].device) for i in range(1, len(inputs))]
                )
                
            else: attr = explainer.attribute(
                inputs=inputs, baselines=baselines, target=target,
                additional_forward_args=additional_forward_args
            )
        else: attr = explainer.attribute(
            inputs=inputs, baselines=baselines, target=target,
            additional_forward_args=additional_forward_args
        )
        
        attr_list.append(attr)
    
    if type(inputs) == tuple:
        attr = []
        for input_index in range(len(inputs)):
            attr_per_input = torch.stack([score[input_index] for score in attr_list])
            # pred_len x batch x seq_len x features -> batch x pred_len x seq_len x features
            attr_per_input = attr_per_input.permute(1, 0, 2, 3)
            attr.append(attr_per_input)
            
        attr = tuple(attr)
    else:
        attr = torch.stack(attr_list)
        # pred_len x batch x seq_len x features -> batch x pred_len x seq_len x features
        attr = attr.permute(1, 0, 2, 3)
        
    return attr

def compute_attr(
    name, inputs, baselines, explainer,
    additional_forward_args, args
):
    # name = explainer.get_name()
    if args.task_name == 'classification':
        targets = args.num_class
    else: targets = args.pred_len
        
    if type(inputs) == tuple:
        sliding_window_shapes = tuple([
            (1,1) for _ in inputs
        ])
    else: sliding_window_shapes = (1,1)
    
    if name == 'wtsr':
        attr = explainer.attribute(
            inputs=inputs,
            sliding_window_shapes=sliding_window_shapes,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            threshold=0.5, normalize=True,
            attributions_fn=abs
        )
    
    # these methods need a target specified for multi-output models
    elif name in [
        'deep_lift', 'lime', 'integrated_gradients', 
        'gradient_shap'
    ]:
        attr = compute_attr_with_gradient(
            name, inputs, baselines, explainer,
            additional_forward_args, args, targets
        )

    elif name in ['feature_ablation']:
        attr = explainer.attribute(
            inputs=inputs,
            attributions_fn=abs,
            additional_forward_args=additional_forward_args
        )
        
    elif name in ['feature_permutation', 'winIT']:
        attr = explainer.attribute(
            inputs=inputs, attributions_fn=abs,
            additional_forward_args=additional_forward_args
        )
    elif name =='tsr':
        attr = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            additional_forward_args=additional_forward_args,
            # set the threshold to 0 if you want to avoid skipping any time steps
            threshold=0.55 
        )
    elif name in ['dyna_mask', 'extremal_mask']:
        attr = compute_attr_with_trainer(
            inputs, explainer,
            additional_forward_args, targets
        )
        
    elif name == 'fit':
        if type(inputs) == tuple:
            new_additional_forward_args = tuple([
                input for input in inputs[1:]
            ]) + additional_forward_args
            
            # output is a tuple of length 1, since only one input is used
            attr = explainer.attribute(
                inputs=inputs[0], baselines=baselines[0],
                additional_forward_args=new_additional_forward_args
            )
            
            attr = tuple([attr] + [
                torch.zeros_like_like(inputs[i], device=inputs[i].device) 
                for i in range(1, len(inputs))]
            )
        else: attr = explainer.attribute(
            inputs=inputs,
            additional_forward_args=additional_forward_args
        )
    elif name == 'occlusion':
        attr = explainer.attribute(
            inputs=inputs,
            baselines=baselines,
            sliding_window_shapes = sliding_window_shapes,
            additional_forward_args=additional_forward_args,
            attributions_fn=abs
        )
    elif name=='augmented_occlusion':
        attr = explainer.attribute(
            inputs=inputs,
            sliding_window_shapes = sliding_window_shapes,
            additional_forward_args=additional_forward_args,
            attributions_fn=abs
        )
    elif name == 'gatemask':
        attr = compute_attr_with_gatemask(
            inputs, baselines, explainer,
            additional_forward_args, args, targets
        )
    else:
        raise NotImplementedError(f'Explainer {name} is not implemented')
      
    attr = reshape_over_output_horizon(attr, inputs, args)
    attr = round_up(attr, decimals=6)  
    return attr