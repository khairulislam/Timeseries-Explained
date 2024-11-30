import os
import torch
from data.data_factory import data_provider
from models import Autoformer, Transformer, TimesNet, Nonstationary_Transformer, DLinear, FEDformer, \
    Informer, LightTS, Reformer, ETSformer, Pyraformer, PatchTST, MICN, Crossformer, \
    FiLM, LSTM, TCN, SegRNN, iTransformer, TimeLLM, CALF, OFA
from utils.distillationLoss import DistillationLoss

def stringify_setting(args, complete=False):
    if not complete:
        # first two conditions for specific ablations
        if args.task_name == 'classification' and args.seq_len != 48:
            return f"{args.data_path.split('.')[0]}_{args.model}_sl_{args.seq_len}"
        elif args.task_name == 'long_term_forecast' and args.seq_len != 96:
            return f"{args.data_path.split('.')[0]}_{args.model}_sl_{args.seq_len}"
        
        return f"{args.data_path.split('.')[0]}_{args.model}"
    
    setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}'.format(
        args.model,
        args.data_path.split('.')[0],
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads
    )
    
    return setting

dual_input_users = [
    'iTransformer', 'Autoformer', 'ETSformer', 'FEDformer', 
    'Informer', 'Nonstationary_Transformer', 'Reformer', 
    'RNN', 'TimesNet', 'Transformer'
]

class Exp_Basic(object):
    model_dict = {
        'TimesNet': TimesNet,
        'Autoformer': Autoformer,
        'Transformer': Transformer,
        'Nonstationary_Transformer': Nonstationary_Transformer,
        'DLinear': DLinear,
        'FEDformer': FEDformer,
        'Informer': Informer,
        'LightTS': LightTS,
        'Reformer': Reformer,
        'ETSformer': ETSformer,
        'PatchTST': PatchTST,
        'Pyraformer': Pyraformer,
        'MICN': MICN,
        'Crossformer': Crossformer,
        'FiLM': FiLM,
        'LSTM': LSTM,
        'TCN': TCN,
        'SegRNN': SegRNN,
        'iTransformer': iTransformer,
        'TimeLLM': TimeLLM,
        'CALF': CALF,
        'OFA': OFA
    }
    
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        self.setting = stringify_setting(args)
        
        if args.itr_no is not None:
            self.output_folder = os.path.join(
                args.result_path, self.setting, str(args.itr_no)
            )
        else:
            self.output_folder = os.path.join(args.result_path, self.setting)
            
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)
        print(f'Experiments will be saved in {self.output_folder}')
        
        self.dataset_map = {}

    def _build_model(self):
        raise NotImplementedError
    
    def _select_optimizer(self):
        if self.args.model == 'CALF':
            param_dict = [
                {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' in n], "lr": 1e-4},
                {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and '_proj' not in n], "lr": self.args.learning_rate}
            ]
            model_optim = torch.optim.Adam([param_dict[1]], lr=self.args.learning_rate)
            loss_optim = torch.optim.Adam([param_dict[0]], lr=self.args.learning_rate)
            return model_optim, loss_optim
        else:
            model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            return model_optim
    
    def _select_lr_scheduler(self, optimizer):
        if self.args.model in ['CALF', 'OFA']:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.args.tmax, 
                eta_min=1e-8, verbose=True
            )
        else:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=1, factor=0.1, 
                verbose=True, min_lr=5e-6
            )
    
    def load_best_model(self):
        best_model_path = os.path.join(self.output_folder, 'checkpoint.pth')
        print(f'Loading model from {best_model_path}')
        self.model.load_state_dict(torch.load(best_model_path))

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag='test'):
        if flag not in self.dataset_map:
            self.dataset_map[flag] = data_provider(self.args, flag)
            
        return self.dataset_map[flag] 

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
