import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm
import os
import numpy as np
import time
from sklearn.metrics import precision_recall_fscore_support
from storage_utils import save_statistics
import sys
np.set_printoptions(threshold=sys.maxsize)

class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, experiment_name, num_epochs, train_data, valid_data,
                 test_data, weight_decay_coefficient, lr, use_gpu, gpu_id, pretrained, schedule, continue_from_epoch=-1,
                 num_batches=float("inf")):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best valid model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param valid_data: An object of the DataProvider type. Contains the valid set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()
        if torch.cuda.is_available() and use_gpu:  # checks whether a cuda gpu is available and whether the gpu flag is True
            if "," in gpu_id:
                self.device = [torch.device('cuda:{}'.format(idx)) for idx in gpu_id.split(",")]  # sets device to be cuda
            else:
                self.device = torch.device('cuda:{}'.format(gpu_id))  # sets device to be cuda

            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # sets the main GPU to be the one at index 0 (on multi gpu machines you can choose which one you want to use by using the relevant GPU ID)
            print("use GPU")
            print("GPU ID {}".format(gpu_id))
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU

        self.experiment_name = experiment_name
        self.model = network_model
        self.model.reset_parameters()

        print("Num params:", self.get_num_parameters())

        if type(self.device) is list:
            self.model.to(self.device[0])
            self.model = nn.DataParallel(module=self.model, device_ids=self.device)
            self.device = self.device[0]
        else:
            self.model.to(self.device)  # sends the model from the cpu to the gpu
          # re-initialize network parameters
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        self.optimizer = optim.Adam(self.parameters(), lr=lr, amsgrad=False, weight_decay=weight_decay_coefficient)

        if schedule:
            lr_lambda = lambda epoch: 1 if epoch<=0 else 1 / epoch ** 0.5
            optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda, last_epoch=-1)

        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))
        self.weights_folder = os.path.abspath("init_model")

        # Set best models to be at 0 since we are just starting
        self.best_valid_model_idx = 0
        self.best_valid_model_acc = 0.
        self.best_valid_model_f1 = 0.

        self.debug_batch_cnt_max = num_batches

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory

        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory

        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.num_epochs = num_epochs
        # self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU
        if continue_from_epoch == -1:
            self.init_from_scratch(pretrained)
        elif continue_from_epoch == -2:
            try:
                self.best_valid_model_idx, self.best_valid_model_acc, self.best_valid_model_f1, self.state = self.load_model(
                    model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                    model_idx='latest')  # reload existing model from epoch and return best valid model index
                # and the best valid acc of that model
                self.starting_epoch = self.state['current_epoch_idx'] + 1 # start from next epoch
            except:
                print("Model objects cannot be found, initializing a new model and starting from scratch")
                self.init_from_scratch(pretrained)
        else:
            self.best_valid_model_idx, self.best_valid_model_acc, self.best_valid_model_f1, self.state = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best valid model index
            # and the best valid acc of that model
            self.starting_epoch = self.state['current_epoch_idx'] + 1 # start from next epoch


    def init_from_scratch(self, pretrained):
        self.starting_epoch = 0
        if pretrained:
            _,_,_, self.state = self.load_model(
                model_save_dir=self.weights_folder, model_save_name="init_model", model_idx="pretrained")
        else:
            self.state = dict()


    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def save_model(self, model_save_dir, model_save_name, model_idx, state):
        """
        Save the network parameter state and current best valid epoch idx and best valid accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        state['network'] = self.state_dict()  # save network parameter and other variables.
        torch.save(state, f=os.path.join(model_save_dir, "{}_{}.pth".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best valid model idx and best valid acc to be compared with the future valid accuracies, in order to choose the best valid model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best valid idx and best valid model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}.pth".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state['best_valid_model_idx'], state['best_valid_model_acc'], state['best_valid_model_f1'], state


    def reverse_rescale(self, X, min, max):
        """ in-place for minimal RAM consumption """
        ### Meaning: X = min + X * (max-min) / 255
        max -= min
        max /= 255
        X *= torch.unsqueeze(torch.unsqueeze(max, -1), -1)  # add two empty axes for broadcasting
        X += torch.unsqueeze(torch.unsqueeze(min, -1), -1)
        return X

    def run_batch(self, x, x_flow, y, minmax, mode): # mode == True means training mode
        if mode:
            self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)
        else:
            self.eval()  # sets the system to validation mode

        # Expecting a tensor in x, x_flow, and y
        # ss = time.time()
        # to(device=None, dtype=None, non_blocking=False, copy=False) → Tensor
        # Send to CUDA as uint8:
        x = x.to(device=self.device, dtype=torch.uint8, non_blocking=True)
        x_flow = x_flow.to(device=self.device, dtype=torch.uint8, non_blocking=True)
        y = y.to(device=self.device, dtype=torch.uint8, non_blocking=True)
        minmax = minmax.to(device=self.device, dtype=torch.float32, non_blocking=True)
        # print("Sending tensors took {0:.2f} secs".format(time.time() - ss))

        # ss = time.time()
        # Cast to float on CUDA:
        x = x.to(dtype=torch.float)
        x_flow = x_flow.to(dtype=torch.float)
        y = y.to(dtype=torch.long)
        # print("Casting tensors took {0:.2f} secs".format(time.time() - ss))

        # ss = time.time()
        self.reverse_rescale(x_flow, minmax[:, :, 0], minmax[:, :, 1])  # in-place
        # print("Reverse rescale took {0:.2f} secs".format(time.time() - ss))

        del minmax

        # ss = time.time()
        out_spatial, out_temporal = self.model.forward(x, x_flow)  # forward the data in the model
        # print("Forward pass took {0:.2f} secs".format(time.time() - ss))


        # ss = time.time()

        # Get overall loss:
        log_prob_spatial = F.log_softmax(out_spatial, dim=1)
        del out_spatial
        log_prob_temporal = F.log_softmax(out_temporal, dim=1)
        del out_temporal

        log_prob = (log_prob_spatial + log_prob_temporal) / 2  # take the average log-prediction, i.e. the log joint probability

        # print("Requires gradients:",prob_spatial.requires_grad,prob_temporal.requires_grad, log_prob.requires_grad) # True True True
        # print("Average log-prediction:", prob)
        # print("Ground truth:", y)
        # log_softmax + NLL == cross_entropy
        # compute loss:
        loss          = F.nll_loss(input=log_prob, target=y)
        loss_spatial  = F.nll_loss(input=log_prob_spatial, target=y)
        loss_temporal = F.nll_loss(input=log_prob_temporal, target=y)
        # print("Computing probs and losses took {0:.2f} secs".format(time.time() - ss))

        # print(mode)
        if mode:
            # print("optimizing")
            # ss = time.time()
            self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
            loss.backward()  # backpropagate to compute gradients for current iter loss

            # plot_grad_flow_v2(self.model.named_parameters())

            self.optimizer.step()  # update network parameters
            # print("Update took {0:.2f} secs".format(time.time() - ss))

        # get argmax of predictions. indicates which class has highest probability for each sample:
        _, predicted   = torch.max(log_prob.data, 1)
        _, predicted_s = torch.max(log_prob_spatial.data, 1)
        _, predicted_t = torch.max(log_prob_temporal.data, 1)

        accuracy, precision, recall, f1 = self.get_stats(y, predicted)
        a_s, p_s, r_s, f_s              = self.get_stats(y, predicted_s)
        a_t, p_t, r_t, f_t              = self.get_stats(y, predicted_t)

        return loss.data.detach().cpu().numpy(), loss_spatial.data.detach().cpu().numpy(), loss_temporal.data.detach().cpu().numpy(), accuracy, a_s, a_t, precision, p_s, p_t, recall, r_s, r_t, f1, f_s, f_t, y, log_prob.data #[:,1] # log pred prob for class 1

    def run_epoch(self, current_epoch_losses, dataset, output, epoch_idx):
        """ dataset must be one of train, valid, test
             output must be True (if summary.csv should be created) or False (if not)
        """
        if dataset == "train":
            data = self.train_data
            mode = True
        elif dataset == "valid":
            data = self.valid_data
            mode = False
        elif dataset == "test":
            data = self.test_data
            mode = False
        else:
            raise Exception("ERROR: dataset '{}' in run_epoch is wrong!".format(dataset))
            return

        if output:
            y_out = torch.tensor([], dtype=torch.long)  # for ROC curve
            pred_out = torch.tensor([], dtype=torch.float)  # for ROC curve
        else:
            y_out, pred_out = None, None

        debug_batch_cnt = 0
        ss = time.time() # just as initializer
        with tqdm.tqdm(total=len(data)) as pbar:  # create a progress bar
            for idx, (x, x_flow, y, minmax) in enumerate(data):  # get data batches
                # print("")
                # print("{0}: Time between batches took {1:.2f} secs".format(dataset, time.time() - ss))
                if debug_batch_cnt >= self.debug_batch_cnt_max:
                    break
                ss = time.time()
                if mode: # train
                    return_value = self.run_batch(x=x,x_flow=x_flow,y=y,minmax=minmax,mode=mode)
                else: # evaluate
                    with torch.no_grad():  # speeds up and saves memory
                        return_value = self.run_batch(x=x, x_flow=x_flow, y=y, minmax=minmax, mode=mode)

                loss, loss_spatial, loss_temporal, accuracy, a_s, a_t, precision, p_s, p_t, recall, r_s, r_t, f1, f_s, f_t, y, pred = return_value

                # print("{0}: Run {1:.2f} secs".format(dataset, time.time() - ss))
                del x, x_flow, minmax
                current_epoch_losses["{}_loss".format(dataset)].append(loss)  # add current iter loss to the {} loss list
                current_epoch_losses["{}_l_s".format(dataset)].append(loss_spatial)
                current_epoch_losses["{}_l_t".format(dataset)].append(loss_temporal)
                current_epoch_losses["{}_acc".format(dataset)].append(accuracy)  # add current iter acc to the {} acc list
                current_epoch_losses["{}_a_s".format(dataset)].append(a_s)
                current_epoch_losses["{}_a_t".format(dataset)].append(a_t)
                current_epoch_losses["{}_f1".format(dataset)].append(f1)
                current_epoch_losses["{}_f_s".format(dataset)].append(f_s)
                current_epoch_losses["{}_f_t".format(dataset)].append(f_t)
                current_epoch_losses["{}_prec".format(dataset)].append(precision)
                current_epoch_losses["{}_p_s".format(dataset)].append(p_s)
                current_epoch_losses["{}_p_t".format(dataset)].append(p_t)
                current_epoch_losses["{}_rec".format(dataset)].append(recall)
                current_epoch_losses["{}_r_s".format(dataset)].append(r_s)
                current_epoch_losses["{}_r_t".format(dataset)].append(r_t)

                if output:
                    y_out = torch.cat((y_out.cpu(), y.cpu()))
                    pred_out = torch.cat((pred_out.cpu(), pred.cpu()))
                    # print("y_out: {}".format(y_out))
                    # print("pred_out: {}".format(pred_out))

                pbar.update(1)
                pbar.set_description(
                    "E-{:2}: a-{:.3f}, l-{:.3f}, ls-{:.3f}, lt-{:.3f}, f-{:.3f}, p-{:.3f}, r-{:.3f}".format(
                        epoch_idx, accuracy, loss, loss_spatial, loss_temporal, f1, precision, recall))
                # print("{0}ing this batch took {1:.2f} secs".format(dataset, time.time() - ss))

                debug_batch_cnt += 1

                ss = time.time()

        return current_epoch_losses, y_out, pred_out

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best valid model and valid model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_loss": [], "train_l_s": [], "train_l_t": [],
                        "train_acc": [], "train_a_s": [], "train_a_t": [],
                        "train_f1": [], "train_f_s": [], "train_f_t": [],
                        "train_prec": [], "train_p_s": [], "train_p_t": [],
                        "train_rec": [], "train_r_s": [], "train_r_t": [],
                        "valid_loss": [], "valid_l_s": [], "valid_l_t": [],
                        "valid_acc": [], "valid_a_s": [], "valid_a_t": [],
                        "valid_f1": [], "valid_f_s": [], "valid_f_t": [],
                        "valid_prec": [], "valid_p_s": [], "valid_p_t": [],
                        "valid_rec": [], "valid_r_s": [], "valid_r_t": [],
                        "curr_epoch": []} # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.starting_epoch + self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_loss": [], "train_l_s": [], "train_l_t": [],
                                    "train_acc": [], "train_a_s": [], "train_a_t": [],
                                    "train_f1": [], "train_f_s": [], "train_f_t": [],
                                    "train_prec": [], "train_p_s": [], "train_p_t": [],
                                    "train_rec": [], "train_r_s": [], "train_r_t": [],
                                    "valid_loss": [], "valid_l_s": [], "valid_l_t": [],
                                    "valid_acc": [], "valid_a_s": [], "valid_a_t": [],
                                    "valid_f1": [], "valid_f_s": [], "valid_f_t": [],
                                    "valid_prec": [], "valid_p_s": [], "valid_p_t": [],
                                    "valid_rec": [], "valid_r_s": [], "valid_r_t": []}

            training_start_time = time.time()

            print("debug_batch_cnt_max={}".format(self.debug_batch_cnt_max))

            current_epoch_losses, _, _ = self.run_epoch(current_epoch_losses, "train", output=False, epoch_idx=epoch_idx)

            print("Total epoch training time is {0:.2f} secs".format(time.time() - training_start_time))
            validation_start_time = time.time()

            current_epoch_losses, _, _ = self.run_epoch(current_epoch_losses, "valid", output=False, epoch_idx=epoch_idx)


            print("Total epoch validation time is {0:.2f} secs".format(time.time() - validation_start_time))

            # valid_mean_accuracy = np.mean(current_epoch_losses['valid_acc'])
            valid_mean_acc = np.mean(current_epoch_losses['valid_acc'])
            valid_mean_f1 = np.mean(current_epoch_losses['valid_f1'])
            if valid_mean_acc > self.best_valid_model_acc:  # if current epoch's mean valid acc is greater than the saved best valid acc then
                self.best_valid_model_acc = valid_mean_acc  # set the best valid model acc to be current epoch's valid acc
                self.best_valid_model_f1 = valid_mean_f1
                self.best_valid_model_idx = epoch_idx  # set the experiment-wise best valid idx to be the current epoch's idx

            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(
                    value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            total_losses['curr_epoch'].append(epoch_idx)
            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False) # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics

            self.state['current_epoch_idx'] = epoch_idx
            self.state['best_valid_model_acc'] = self.best_valid_model_acc
            self.state['best_valid_model_f1'] = self.best_valid_model_f1
            self.state['best_valid_model_idx'] = self.best_valid_model_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best valid idx and best valid acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx, state=self.state)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best valid idx and best valid acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest', state=self.state)

            print("Epoch {}: {}".format(epoch_idx, out_string))
            print("Total epoch time is {0:.2f} secs".format(time.time() - epoch_start_time))

        valid_losses = self.output_summary(type='valid', epoch_idx=epoch_idx)
        test_losses = self.output_summary(type='test', epoch_idx=epoch_idx)

        return total_losses, test_losses

    # helper function
    def output_summary(self, type, epoch_idx):
        print("Generating {} set evaluation metrics".format(type))

        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_valid_model_idx,
                        # load best validation model
                        model_save_name="train_model")
        current_epoch_losses = {type + "_loss": [], type+"_l_s": [], type+"_l_t": [],
                                type + "_acc": [], type+"_a_s": [], type+"_a_t": [],
                                type + "_f1": [], type+"_f_s": [], type+"_f_t": [],
                                type + "_prec": [], type + "_p_s": [], type + "_p_t": [],
                                type + "_rec": [], type+"_r_s": [], type+"_r_t": []} # initialize a statistics dict


        current_epoch_losses, y_out, pred_out = self.run_epoch(current_epoch_losses, type, output=True, epoch_idx=epoch_idx)

        # save test set metrics in dict format
        losses = {}
        losses[type+"_loss"] = [np.mean(current_epoch_losses[type+"_loss"])]
        losses[type + "_l_s"] = [np.mean(current_epoch_losses[type + "_l_s"])]
        losses[type + "_l_t"] = [np.mean(current_epoch_losses[type + "_l_t"])]
        losses[type + "_acc"] = [np.mean(current_epoch_losses[type + "_acc"])]
        losses[type + "_a_s"] = [np.mean(current_epoch_losses[type + "_a_s"])]
        losses[type + "_a_t"] = [np.mean(current_epoch_losses[type + "_a_t"])]
        losses[type + "_f1"] = [np.mean(current_epoch_losses[type + "_f1"])]
        losses[type + "_f_s"] = [np.mean(current_epoch_losses[type + "_f_s"])]
        losses[type + "_f_t"] = [np.mean(current_epoch_losses[type + "_f_t"])]
        losses[type + "_prec"] = [np.mean(current_epoch_losses[type + "_prec"])]
        losses[type + "_p_s"] = [np.mean(current_epoch_losses[type + "_p_s"])]
        losses[type + "_p_t"] = [np.mean(current_epoch_losses[type + "_p_t"])]
        losses[type + "_rec"] = [np.mean(current_epoch_losses[type + "_rec"])]
        losses[type + "_r_s"] = [np.mean(current_epoch_losses[type + "_r_s"])]
        losses[type + "_r_t"] = [np.mean(current_epoch_losses[type + "_r_t"])]



        y_out = y_out.detach().cpu().numpy().flatten()
        # losses[type+'_y'] = [str(y_arr)]
        pred_out = pred_out.detach().cpu().numpy() #.flatten()
        # pred_out = np.exp(pred_arr) # pred_arr is log probabilities, make probabilities
        # losses[type+'_pred'] = [str(self.sigmoid_array(pred_arr))]

        print(type + " evaluation:")
        print("shape of y:   ", y_out.shape)
        print("shape of pred:", pred_out.shape)

        out_path = os.path.join(self.experiment_logs, "{}_out.npz".format(type))
        # print("Saving in file {}: y_out {}, and pred_out {}".format(out_path, y_out, pred_out))
        np.savez(out_path, y=y_out, pred=pred_out)


        save_statistics(experiment_log_dir=self.experiment_logs, filename=type+'_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=losses, current_epoch=0, continue_from_mode=False)
        return losses

    # helper function
    def get_stats(self, y, predicted):
        accuracy = np.mean(list(predicted.eq(y.data).cpu()))  # compute accuracy

        # Might throw UndefinedMetricWarning if current batch contains only samples of one single class
        precision, recall, f1, _ = np.array(precision_recall_fscore_support(y.data.cpu().numpy(), predicted.cpu().numpy()))[:,-1]
        return accuracy, precision, recall, f1
