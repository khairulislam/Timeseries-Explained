from data.data_factory import data_provider
from exp.exp_basic import *
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch, os, time, warnings
import numpy as np
import pandas as pd
from os.path import join
from datetime import datetime

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        initializer = self.model_dict[self.args.model]
        
        if self.args.model in ['CALF', 'OFA', 'MICN']:
            model = initializer.Model(self.args, self.device).float()
        else:
            model = initializer.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_criterion(self):
        if self.args.model == 'CALF':
            criterion = DistillationLoss(
                self.args.distill_loss, 
                self.args.logits_loss, 
                self.args.task_loss, 
                self.args.task_name, 
                self.args.feature_w, 
                self.args.logits_w, 
                self.args.task_w,
                self.args.features, 
                self.args.pred_len
            )
        elif self.args.model == 'OFA':
            criterion = torch.nn.L1Loss()
        else: criterion = torch.nn.MSELoss()
        
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        if self.args.model == 'CALF':
            self.model.in_layer.eval()
            self.model.out_layer.eval()
            self.model.time_proj.eval()
            self.model.text_proj.eval()
            
            criterion = torch.nn.MSELoss()
        elif self.args.model == 'OFA':
            self.model.in_layer.eval()
            self.model.out_layer.eval()
        else:
            self.model.eval()
            
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.model == 'CALF':
                    outputs = self.model(batch_x)['outputs_time']
                elif self.args.model == 'OFA':
                    outputs = self.model(batch_x)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        
        if len(total_loss) == 0:
            print('Warning: no loss values found.')
            total_loss = np.inf
        else:
            total_loss = np.average(total_loss)
        
        if self.args.model == 'CALF':
            self.model.in_layer.train()
            self.model.out_layer.train()
            self.model.time_proj.train()
            self.model.text_proj.train()
        elif self.args.model == 'OFA':
            self.model.in_layer.train()
            self.model.out_layer.train()
        else: 
            self.model.train()
        return total_loss

    def train(self):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')

        path = self.output_folder

        time_now = time.time()
        start = datetime.now()
        print(f'Training started at {start}.')

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.args.model == 'CALF':
            model_optim, loss_optim = self._select_optimizer()
        else: model_optim = self._select_optimizer()
            
        criterion = self._select_criterion()
        lr_scheduler = self._select_lr_scheduler(model_optim)
        f_dim = -1 if self.args.features == 'MS' else 0

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                if self.args.model == 'CALF': 
                    loss_optim.zero_grad()
                    
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.model in ['CALF', 'OFA']:
                    outputs = self.model(batch_x)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # only CALF model has dictionary output
                if self.args.model != 'CALF':
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    if self.args.model == 'CALF':
                        loss_optim.zero_grad()

            print(f"Epoch: {epoch + 1} cost time: { time.time() - epoch_time}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.5g} Vali Loss: {vali_loss:.5g}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            elif early_stopping.counter > 0:
                # resume training from last best model
                self.load_best_model()

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        end = datetime.now()
        epochs_run = epoch + early_stopping.early_stop + 1
        print(f'Training ended at {end}, time taken {end-start}, per epoch {(end-start)/epochs_run}')
        self.load_best_model()
        return self.model

    def test(
        self, load_model:bool=False, flag:str='test'
    ):
        _, test_loader = self._get_data(flag=flag)
        if load_model: self.load_best_model()

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                
                if self.args.model == 'CALF':
                    outputs = self.model(batch_x)['outputs_time']
                elif self.args.model == 'OFA':
                    outputs = self.model(batch_x)
                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)
                

        # this line handles different size of batch. E.g. last batch can be < batch_size.
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('Preds and Trues shape:', preds.shape, trues.shape)

        mae, mse, rmse = metric(preds, trues)
        result_string = f'flag:{flag}, mse:{mse:0.5g}, mae:{mae:0.5g}, rmse: {rmse:0.5g}.'
        print(result_string)
        f = open(os.path.join(self.args.result_path, "result_long_term_forecast.txt"), 'a')
        f.write(stringify_setting(self.args, complete=True) + "  \n")
        f.write(result_string + '\n\n')
        f.close()
        
        results = pd.DataFrame({
            'metric': ['mae', 'mse', 'rmse'], 
            'score':[mae, mse, rmse]
        })

        results.to_csv(join(self.output_folder, f'{flag}_metrics.csv'), index=False)
        np.save(join(self.output_folder, f'{flag}_pred.npy'), preds)
        np.save(join(self.output_folder, f'{flag}_true.npy'), trues)

        return
