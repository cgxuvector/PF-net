"""
    Implementation of Particle Filters Network:
        - The Spatial Transformer Network is used to crop local maps from the global map.
        - We can make an assumption to make the set up easier, which is the states has access to the map.
        - To be aligned, there is no noise
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationModel(nn.Module):
    def __init__(self, obs_type="color", batch_size=32, particle_num=4):
        super(ObservationModel, self).__init__()

        # observation type
        self.obs_type = obs_type

        # set observation size
        self.obs_width = 56
        self.obs_height = 56

        # set local map size
        self.local_map_width = 28
        self.local_map_height = 28

        # set the particle num
        self.k = particle_num

        # batch size
        self.batch_size = batch_size

        # observation encoder
        # batch x 3 x 56 x 56
        self.color_obs_encoder = nn.Sequential(
            # 3 x 56 x 56 -> 64 x 28 x 28
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 64 x 28 x 28 -> 128 x 14 x 14
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # 128 x 14 x 14 -> 64 x 14 x 14
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),

            # 64 x 14 x 14 -> 32 x 14 x 14
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),

            # 32 x 14 x 14 -> 16 x 14 x 14
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1)),
            nn.ReLU(inplace=True)
        )

        self.depth_obs_encoder = nn.Sequential(
            # 1 x 56 x 56 -> 64 x 28 x 28
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            # 64 x 28 x 28 -> 128 x 14 x 14
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True),

            # 128 x 14 x 14 -> 64 x 14 x 14
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),

            # 64 x 14 x 14 -> 32 x 14 x 14
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)),
            nn.ReLU(inplace=True),

            # 32 x 14 x 14 -> 16 x 14 x 14
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1)),
            nn.ReLU(inplace=True)
        )

        # map encoder
        # 1 x 28 x 28
        self.map_encoder = nn.Sequential(
            # 1 x 28 x 28 -> 8 x 14 x 14
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )

        # compress the channel
        self.local_connect_conv = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=16, kernel_size=(5, 5)),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(inplace=True)
        )

        # fully connected layer
        self.fc_layer = nn.Sequential(
            nn.Linear(5 * 5 * 16, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, o, m):
        # encode the observation
        if self.obs_type == "color":
            # convert obs: batch x k x 3 x H x W -> (batch * k) x 3 x H x W
            o = torch.reshape(o, (-1, 1, 3, self.obs_height, self.obs_width)).squeeze(dim=1)
            o_feature = self.color_obs_encoder(o)  # encode RGB observation
        elif self.obs_type == "depth":
            # convert obs: batch x k x 1 x H x W -> (batch * k) x 1 x H x W
            o = torch.reshape(o, (-1, 1, 1, self.obs_height, self.obs_width)).squeeze(dim=1)
            o_feature = self.depth_obs_encoder(o)  # encode Depth observation
        else:
            raise Exception("Wrong observation type. Expect color or depth.")

        # convert map: batch x k x H x W -> (batch * k) x 1 x H x W
        m = torch.reshape(m, (-1, 1, 1, self.local_map_height, self.local_map_width)).squeeze(dim=1)
        # encode the map observation
        m_feature = self.map_encoder(m)  # encode the local map

        # concatenate the channels
        o_m_feature = torch.cat([o_feature, m_feature], dim=1)  # merge through depth channels

        # compress the channels: Note: here locally fully-connected layers are needed but I don't know what is it
        # I just use a simple convolutional layer
        o_m_feature = self.local_connect_conv(o_m_feature).view(self.batch_size, self.k, -1)

        # compute the likelihood
        o_likelihood = self.fc_layer(o_m_feature).view(self.batch_size, -1)

        return o_likelihood


class TransitionModel(nn.Module):
    def __init__(self, batch_size=32):
        super(TransitionModel, self).__init__()

        self.batch_size = batch_size

        self.dyna_model = nn.Sequential(
            nn.Linear(7 + 1, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 7)
        )

    def forward(self, x, u):
        # concatenate the data
        x_u_tensor = torch.cat([x, u], dim=2)
        # predict the offset
        delta = self.dyna_model(x_u_tensor)
        # predict the next state
        next_x = x + delta
        return next_x


class STNet(nn.Module):
    def __init__(self, batch_size=32):
        # inherit from nn
        super(STNet, self).__init__()

        # set the batch size
        self.batch_size = batch_size

        # localization network
        self.localization = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([0.28, 0, 0, 0, 0.28, 0], dtype=torch.float))  # ~(32 / 100)

    # Spatial transformer network forward function
    def stn(self, s, gm):
        # predict the transformation parameters using states
        xs = self.localization(s)
        theta = self.fc_loc(xs).view(-1, 1, 6).squeeze(dim=0)

        theta = theta.view(-1, 2, 3)  # resize for grid sampling

        # give more constraint and resize to batch_size x 2 x 3
        # only contains same scale and translation
        # scale = theta[:, 0].unsqueeze(1)
        # scale_mat = torch.cat((scale, scale), 1)
        # translation = theta[:, 1:].unsqueeze(2)
        # theta = torch.cat((torch.diag_embed(scale_mat), translation), 2)

        # generate the grid sampler usign the affine parameters and the target local map size
        grid = F.affine_grid(theta, torch.Size([theta.size()[0], 1, 28, 28]), align_corners=True)

        # crop the local map from the global map
        gm = gm.expand(theta.size()[0], -1, -1, -1)
        wrap_lm = F.grid_sample(gm, grid, align_corners=True).view(self.batch_size, -1, 28, 28)

        return wrap_lm

    def forward(self, x, gm):
        # Spatial Neural Network
        wrap_lm = self.stn(x, gm)

        return wrap_lm


class PFNet(nn.Module):
    def __init__(self, device, obs_type="color", batch_size=32, particle_num=4, enable_re_sample=False):
        super(PFNet, self).__init__()

        # set the batch size
        self.batch_size = batch_size

        # spatial transformer network
        self.stn_net = STNet(batch_size).to(device)

        # observation model
        self.obs_func = ObservationModel(obs_type=obs_type,
                                         batch_size=batch_size,
                                         particle_num=particle_num).to(device)

        # transition function
        self.trans_func = TransitionModel().to(device)

        # particles
        self.k = particle_num
        self.particles = torch.randn((self.k, 7)).unsqueeze(dim=0).expand(self.batch_size, -1, -1).to(device)
        self.weights = (torch.ones(self.k) * (1.0 / self.k)).expand(self.batch_size, -1).to(device)

        # set the alpha
        self.alpha = 0.5

        # set re-sample
        self.re_sample = enable_re_sample

    def resample(self):
        pass

    def forward(self, o, u, gm):
        """ PF-net forward pass"""
        """ Expand the input """
        # expand the action: batch x 1 -> batch x k x 1
        u = u.unsqueeze(dim=1).expand(-1, self.k, -1)

        # expand the observation: batch x k x 3 x 56 x 56
        o = o.unsqueeze(dim=1).expand(-1, self.k, -1, -1, -1)

        """ Observation update """
        # transform the global map into local maps
        local_m = self.stn_net(self.particles, gm)
        # observation update
        obs_likelihood = self.obs_func(o, local_m)
        self.weights += obs_likelihood  # log space, unnormalized

        """ Re-sample strategy """
        if self.re_sample:
            self.resample()

        # motion update: only affects the particle state input
        self.particles = self.trans_func(self.particles, u)

        # create new state
        tmp_weights = self.weights.unsqueeze(dim=-1).expand(self.batch_size, self.k, 7)
        next_state = torch.mul(self.particles, tmp_weights).sum(dim=1)

        return next_state


# # device
# device = torch.device("cuda:0")
#
# batch_size = 32
#
# # test code
# test_o = torch.randn((batch_size, 3, 56, 56)).to(device)
# test_u = torch.randn((batch_size, 1)).to(device)
# test_m = torch.randn((1, 1, 469, 776)).to(device)
#
# # test model
# test_pf_net = PFNet(device, batch_size=batch_size, particle_num=30)
#
# # output
# test_output = test_pf_net(test_o, test_u, test_m)
# print(test_output)
#
# torch.save(test_pf_net.state_dict(), "test.pt")
#
# batch_size = 1
#
# # test code
# test_o = torch.randn((batch_size, 3, 56, 56)).to(device)
# test_u = torch.randn((batch_size, 1)).to(device)
# test_m = torch.randn((1, 1, 469, 776)).to(device)
#
# test_pf_net_1 = PFNet(device, batch_size=1, particle_num=1000)
# test_pf_net_1.load_state_dict(torch.load("test.pt", map_location=device))
# test_pf_net_1.eval()
#
# print(test_pf_net_1(test_o, test_u, test_m))


        # # update the weight
        # self.weights = torch.mul(self.weights, obs_likelihood)
        # self.weights = self.weights / self.weights.sum()
        #
        # # soft sampling
        # weight_q = self.alpha * self.weights + (1 - self.alpha) * torch.ones_like(self.weights) / self.k
        #
        # # update the weights using soft-sampling
        # self.weights = torch.div(self.weights, weight_q)