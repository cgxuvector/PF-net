import numpy as np
import random
from utils.Params import ParamsLoader
import os
import tqdm
import torch
from models.networks import PFNet
from torch import distributions
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import IPython.terminal.debugger as Debug


ACTION_NAME = ["forward", "turn_right", "turn_left"]


def convert_episodes_to_sequence(episodes_data, step_num=1):
    results = []
    for episode in episodes_data:
        # step size
        stride = 2 * step_num
        # loop the episode
        for i in range(0, len(episode)-1, stride):
            tmp_list = []
            # save the sequence as: [o_t, s_t, a_t, o_{t+1}, s_{t+1}, ...., o_{t+stride}, s_{t+stride}]
            for s in range(stride + 1):
                if isinstance(episode[i+s], list):
                    tmp_list.extend(episode[i+s])
                else:
                    tmp_list.append(ACTION_NAME.index(episode[i+s]))
            # save the sequence truck
            results.append(tmp_list)

    return results


def load_maze_data(maze_id, step_num):
    # load all episodes in one maze
    data = np.load(f"/mnt/data/cheng_results/maze_data/maze_7/maze_7_rgb_id_{maze_id}.npy", allow_pickle=True).tolist()

    # map data
    map_data = data[0]

    # convert the episode to step_sum length sequence truck
    seq_data = convert_episodes_to_sequence(episodes_data=data[1], step_num=step_num)

    return map_data, seq_data


def sample_mini_batch(data_set, batch_size):
    # sample the mini-batch data
    sampled_data = random.sample(data_set, batch_size)

    # convert to tensor
    tmp = []
    for i in range(len(sampled_data[0])):
        # merge the data from samples in the batch
        tmp_elem = np.array([d[i] for d in sampled_data])
        # change the shape if it is image
        if len(tmp_elem.shape) == 4:
            tmp_elem = torch.tensor(tmp_elem.transpose(0, 3, 1, 2)).float()
        elif len(tmp_elem.shape) == 2:
            tmp_elem = torch.tensor(tmp_elem).float()
        elif len(tmp_elem.shape) == 1:
            tmp_elem = torch.tensor(tmp_elem).view(-1, 1).float()
        else:
            raise Exception("Wrong data shape")
        # convert to be tensor
        tmp.append(tmp_elem)

    return tmp


def run_val(model, particle_generator, configs):
    # set device
    device = torch.device(configs['device'])

    with torch.no_grad():
        # store the error list
        error_list = []
        # training iteration per epoch
        for v_n in range(configs['val_num']):
            # sample a maze environment
            val_maze_id = random.sample(range(10, 20, 1), 1)[0]
            # load the offline data for that maze
            val_maze_map_data, val_maze_maze_data = load_maze_data(maze_id=val_maze_id,
                                                                   step_num=configs['step_num_per_seq'])
            maze_map_tensor = torch.tensor(val_maze_map_data).float().unsqueeze(dim=0).unsqueeze(dim=0).to(device)

            # sample a mini-batch from the maze data
            val_batch_data = sample_mini_batch(val_maze_maze_data, batch_size=configs['batch_size'])

            # generate particles based on states with 0 mean and a diagonal covariance matrix
            val_states = val_batch_data[1].unsqueeze(dim=1).expand(configs['batch_size'], configs['particle_num'], -1)
            particles_offset = particle_generator.sample(
                (configs['batch_size'] * configs['particle_num'],)).unsqueeze(dim=1)
            val_states = val_states + particles_offset.view(configs['batch_size'], configs['particle_num'], -1)
            val_states = val_states.to(device)
            val_weights = torch.ones((configs['batch_size'],
                                      configs['particle_num'])).float().to(device) / configs['particle_num']
            val_weights = val_weights.to(device)

            # forward pass in the sequence trunk
            cumulative_error = torch.tensor(0.0, requires_grad=True).to(device)
            for i in range(0, len(val_batch_data) - 2, 3):
                # current observation and state
                o_tensor = val_batch_data[i].to(device)
                # action
                a_tensor = val_batch_data[i + 2].to(device)
                # ground truth next state
                gt_next_s_tensor = val_batch_data[i + 4].to(device)

                # forward pass
                pred_next_s_tensor, [val_states, val_weights] = model(o_tensor,
                                                                      a_tensor,
                                                                      maze_map_tensor,
                                                                      [val_states, val_weights])

                # compute the error
                loss = F.mse_loss(pred_next_s_tensor, gt_next_s_tensor)
                cumulative_error += loss

            # save the error
            error_list.append((cumulative_error/configs['step_num_per_seq']).item())

    return np.mean(error_list)


def run_train(configs):
    # create the tensorboard
    tb = SummaryWriter(comment=f"_lr_{configs['lr']}"
                               f"_bs_{configs['batch_size']}"
                               f"_pn_{configs['particle_num']}"
                               f"_sn_{configs['step_num_per_seq']}"
                               f"_cs_{configs['covariance_scale']}")

    # device
    device = torch.device(configs['device'])

    # create the PF-net model
    pf_net = PFNet(state_dim=configs['state_dim'],
                   device=device,
                   batch_size=configs['batch_size'],
                   particle_num=30).to(device)

    # define the optimizer
    optimizer = torch.optim.Adam(params=pf_net.parameters(),
                                 lr=configs['lr'],
                                 weight_decay=configs['weight_decay'])

    # define the loss criterion
    criterion = torch.nn.MSELoss()

    # define the mean
    mean_tensor = torch.zeros((configs['state_dim']))
    covariance_matrix_tensor = torch.eye(configs['state_dim']) * configs['covariance_scale']
    particle_generator = distributions.MultivariateNormal(mean_tensor, covariance_matrix_tensor)

    # record the last evaluation
    last_error = np.inf

    # training epochs
    for ep_id, ep in enumerate(tqdm.tqdm(range(configs["epoch_num"]),
                                         desc="Train epoch loop",
                                         position=0)):
        # sample a maze environment
        maze_id = random.sample(range(0, 10, 1), 1)[0]
        # load the offline data for that maze
        maze_map_data, maze_maze_data = load_maze_data(maze_id=maze_id,
                                                       step_num=configs['step_num_per_seq'])
        maze_map_tensor = torch.tensor(maze_map_data).float().unsqueeze(dim=0).unsqueeze(dim=0).to(device)
        # training iteration per epoch
        for it_id, it in enumerate(tqdm.tqdm(range(configs["iter_num_per_epoch"]),
                                   desc=f"{ep_id} | iteration loop",
                                   position=1,
                                   leave=False)):
            # sample a mini-batch from the maze data
            batch_data = sample_mini_batch(maze_maze_data, batch_size=configs['batch_size'])

            # generate particles based on states with 0 mean and a diagonal covariance matrix
            state_tensor = batch_data[1].unsqueeze(dim=1).expand(configs['batch_size'], configs['particle_num'], -1)
            particles_offset = particle_generator.sample((configs['batch_size'] * configs['particle_num'],)).unsqueeze(dim=1)
            states = state_tensor + particles_offset.view(configs['batch_size'], configs['particle_num'], -1)
            states = states.to(device)
            weights = torch.ones((configs['batch_size'],
                                  configs['particle_num'])).float().to(device) / configs['particle_num']
            weights = weights.to(device)

            # forward pass in the sequence trunk
            cumulative_error = torch.tensor(0.0, requires_grad=True).to(device)
            for i in range(0, len(batch_data) - 2, 3):
                # current observation and state
                o_tensor = batch_data[i].to(device)
                # action
                a_tensor = batch_data[i + 2].to(device)
                # ground truth next state
                gt_next_s_tensor = batch_data[i + 4].to(device)

                # forward pass
                pred_next_s_tensor, [states, weights] = pf_net(o_tensor, a_tensor, maze_map_tensor, [states, weights])

                # compute the error
                loss = criterion(pred_next_s_tensor, gt_next_s_tensor)
                cumulative_error = cumulative_error + loss

            # back propagation
            optimizer.zero_grad()
            cumulative_error.backward()
            optimizer.step()

            # save the best performance model based on evaluation
            if (ep * configs['iter_num_per_epoch'] + it + 1) % configs['val_iter_freq'] == 0:
                val_error = run_val(pf_net, particle_generator, configs)
                if val_error < last_error:
                    torch.save(pf_net.state_dict(),
                               f"{configs['save_dir']}/pf_net"
                               f"_lr_{configs['lr']}"
                               f"_bs_{configs['batch_size']}"
                               f"_pn_{configs['particle_num']}"
                               f"_sn_{configs['step_num_per_seq']}"
                               f"_cs_{configs['covariance_scale']}_model_best.pt")
                    last_error = val_error

            # periodically save the models
            if (ep * configs['iter_num_per_epoch'] + it + 1) % configs['save_period'] == 0:
                torch.save(pf_net.state_dict(),
                           f"{configs['save_dir']}/pf_net"
                           f"_lr_{configs['lr']}"
                           f"_bs_{configs['batch_size']}"
                           f"_pn_{configs['particle_num']}"
                           f"_sn_{configs['step_num_per_seq']}"
                           f"_cs_{configs['covariance_scale']}"
                           f"_model_period_{ep * configs['iter_num_per_epoch'] + it + 1}.pt")

            # plot the training loss tensorboard
            tb.add_scalar("Train MSE Loss", cumulative_error.item(), ep * configs['iter_num_per_epoch'] + it)
            if last_error == np.inf:
                continue
            else:
                tb.add_scalar("Val MSE Loss", last_error, ep * configs['iter_num_per_epoch'] + it)


if __name__ == "__main__":
    # parse the input arguments
    run_configs = ParamsLoader("./Params/run_pf_train_params.json").params_data

    # make the directories to save data
    if not os.path.exists(run_configs['save_dir']):
        os.makedirs(run_configs['save_dir'])

    # set the seed
    random.seed(run_configs['rnd_seed'])
    np.random.seed(run_configs['rnd_seed'])
    torch.manual_seed(run_configs['rnd_seed'])

    # run the training
    run_train(run_configs)



