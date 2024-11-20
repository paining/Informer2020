from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack, InformerStack_layerfeat

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm, trange

import os
import time

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            #'informerstack':InformerStack,
            'informerstack':InformerStack_layerfeat,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers # default e_layers is 2
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; 
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; 
            Data = Dataset_Pred
        else: # flag == 'train
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; 
        data_set = Data(
            root_path=args.root_path, # './data/data_1025/result_1025'
            flag=flag, 
            size=[args.seq_len, args.label_len, args.pred_len], # [100, 50, 20]
            features=args.features,
            target=args.target, # default is 'OT'
            inverse=args.inverse, # default is False
            cols=args.cols
        )
        print("Total Dataset:", len(data_set))
        train_size = int(len(data_set) * 0.7)
        val_size = int(len(data_set) * 0.2)
        test_size = len(data_set) - train_size - val_size

        train_dataset, valid_dataset, test_dataset = random_split(data_set, [train_size, val_size, test_size])
        if (not os.path.exists("data/train_set.npz") and 
            not os.path.exists("data/val_set.npz") and
            not os.path.exists("data/test_est.npz")):
            idxs = train_dataset.indices
            np.savez(
                "data/train_set.npz", 
                X=data_set.final_data[idxs], 
                y=data_set.final_data_tgt[idxs],
                min=data_set.scaler.min,
                max=data_set.scaler.max,
                )
            np.savez(
                "data/val_set.npz", 
                X=data_set.final_data[idxs], 
                y=data_set.final_data_tgt[idxs],
                min=data_set.scaler.min,
                max=data_set.scaler.max,
                )
            np.savez(
                "data/test_set.npz", 
                X=data_set.final_data[idxs], 
                y=data_set.final_data_tgt[idxs],
                min=data_set.scaler.min,
                max=data_set.scaler.max,
                )
        if flag == 'train':
            print("Train Segments:", len(train_dataset))
            data_set = train_dataset
        elif flag == 'val':
            print("Valid Segments:", len(valid_dataset))
            data_set = valid_dataset
        elif flag == 'test':
            print("Test Segments:", len(test_dataset))
            data_set = test_dataset
       
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y) in enumerate(vali_loader):
            pred, true = self._process_one_batch(vali_data, batch_x, batch_y)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path): 
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        losses = {
            "train": [],
            "valid": [],
            "test": [],
        }

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in trange(0, self.args.train_epochs, desc="Train Epoch"):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad() 
                pred, true = self._process_one_batch(train_data, batch_x, batch_y) # batch_x [B, 100, 2], batch_y [B, 50, 2]
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
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

            print()
            print("Epoch: {} (Cost time: {})".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            losses["train"].append(train_loss.item())
            losses["valid"].append(vali_loss.item())
            losses["test"].append(test_loss.item())

            np.savez("losses.npz", **losses)
            fig, ax = plt.subplots()
            for k, v in losses.items():
                ax.plot(v, label=f"{k}", alpha=0.8)
            ax.grid(True, "major", "both", alpha=0.3)
            ax.minorticks_on()
            ax.grid(True, "minor", "both", alpha=0.1)
            ax.set_title("Loss Graph")
            ax.set_xlabel("Training Epoch")
            ax.set_ylabel("Loss")
            fig.tight_layout()
            fig.savefig("loss.png")
            plt.close(fig)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}, Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y) in enumerate(test_loader):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        # decoder input
        zero_input = torch.zeros(batch_y.shape[0], batch_y.shape[1], batch_x.shape[-1]).float().to(self.device)                # [B, 50, 2]
        dec_inp = torch.cat((batch_x[:, batch_x.shape[1]-self.args.label_len:, :], zero_input), dim=1).float().to(self.device) # [B, 100, 2]

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, dec_inp)[0]
                else:
                    outputs = self.model(batch_x, dec_inp)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, dec_inp)[0] # [B, 50, 1]
            else:
                outputs = self.model(batch_x, dec_inp)

        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return outputs, batch_y
