import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.architecture import ResnetSPADEBlock
from torch.utils.checkpoint import checkpoint


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='most',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super(SPADEGenerator, self).__init__()
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        opt.norm_G = 'spectralspadesyncbatch3x3'
        self.opt = opt
        nf = opt.ngf
        activation = nn.ReLU(False)
        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2
            self.up_4_2 = nn.Sequential(norm_layer(nn.ConvTranspose2d(1 * nf, nf // 2,
                                                             kernel_size=3, stride=2,
                                                             padding=1, output_padding=1)),
                               activation)

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        # if self.opt.num_upsampling_layers == 'more' or \
        #    self.opt.num_upsampling_layers == 'most':
        #     x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4_2(x)
            # x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class Interpolate(nn.Module):
    def __init__(self, scale_factor=2, size=None, mode='bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        if self.size is not None:
            x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        else:
            x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x


class Face2FaceGenerator(BaseNetwork):
        @staticmethod
        def modify_commandline_options(parser, is_train):
            parser.add_argument('--resnet_n_downsample', type=int, default=4,
                                help='number of downsampling layers in netG')
            parser.add_argument('--resnet_n_blocks', type=int, default=9,
                                help='number of residual blocks in the global generator network')
            parser.add_argument('--resnet_kernel_size', type=int, default=3,
                                help='kernel size of the resnet block')
            parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                                help='kernel size of the first convolution')
            parser.set_defaults(norm_G='spectralinstance')
            return parser

        def __init__(self, opt):
            super(Face2FaceGenerator, self).__init__()
            input_nc = 3

            norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
            activation = nn.ReLU(False)
            # initial conv
            self.first_layer = nn.Sequential(nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                                             norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                                                  kernel_size=opt.resnet_initial_kernel_size,
                                                                  padding=0)),
                                                                  activation)
            # downsample
            downsample_model = []

            mult = 1
            for i in range(opt.resnet_n_downsample):
                downsample_model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                                          kernel_size=3, stride=2, padding=1)),
                                                          activation]
                mult *= 2

            self.downsample_layers = nn.Sequential(*downsample_model)

            # resnet blocks
            resnet_model = []

            for i in range(opt.resnet_n_blocks):
                resnet_model += [ResnetBlock(opt.ngf * mult,
                                             norm_layer=norm_layer,
                                             activation=activation,
                                             kernel_size=opt.resnet_kernel_size)]

            self.resnet_layers = nn.Sequential(*resnet_model)

            # upsample

            upsample_model = []

            for i in range(opt.resnet_n_downsample):
                nc_in = int(opt.ngf * mult)
                nc_out = int((opt.ngf * mult) / 2)
                upsample_model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                                 kernel_size=3, stride=2,
                                                                 padding=1, output_padding=1)),
                                   activation]
                mult = mult // 2

            self.upsample_layers = nn.Sequential(*upsample_model)

            # final output conv
            self.final_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                             nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                                             nn.Tanh())

        def forward(self, input, z=None):
            net = self.first_layer(input)
            net = self.downsample_layers(net)
            net = self.resnet_layers(net)
            net = self.upsample_layers(net)
            net = self.final_layer(net)
            return net


class MS1MGenerator(Face2FaceGenerator):
    def __init__(self, opt):
        super(MS1MGenerator, self).__init__(opt)


class RotateGenerator(Face2FaceGenerator):
    def __init__(self, opt):
        super(RotateGenerator, self).__init__(opt)


class RotateSPADEGenerator(Face2FaceGenerator):
    def __init__(self, opt):
        super(RotateSPADEGenerator, self).__init__(opt)
        del self.resnet_layers
        self.resnet_n_blocks = opt.resnet_n_blocks
        mult = 1
        for i in range(opt.resnet_n_downsample):
            mult *= 2
        for i in range(opt.resnet_n_blocks):
            self.add_module('resnet_layers' + str(i), ResnetSPADEBlock(opt.ngf * mult, opt.semantic_nc))

    def forward(self, input, seg=None):
        # net = self.first_layer(input)
        net = checkpoint(self.first_layer, input)
        # net = self.downsample_layers(net)
        net = checkpoint(self.downsample_layers, net)
        for i in range(self.resnet_n_blocks):
            # net = self._modules['resnet_layers' + str(i)](net, seg)
            net = checkpoint(self._modules['resnet_layers' + str(i)], net, seg)
        # net = self.upsample_layers(net)
        net = checkpoint(self.upsample_layers, net)
        # net = self.final_layer(net)
        net = checkpoint(self.final_layer, net)
        return net


class RotateSPADEBGGenerator(Face2FaceGenerator):
    def __init__(self, opt):
        super(RotateSPADEBGGenerator, self).__init__(opt)
        del self.resnet_layers
        self.resnet_n_blocks = opt.resnet_n_blocks
        mult = 1
        for i in range(opt.resnet_n_downsample):
            mult *= 2
        for i in range(opt.resnet_n_blocks):
            if i == 0:
                self.add_module('resnet_layers' + str(i), ResnetSPADEBlock(opt.ngf * mult, opt.semantic_nc))
            else:
                self.add_module('resnet_layers' + str(i), ResnetSPADEBlock(opt.ngf * mult, opt.label_nc))

    def forward(self, input, seg=None):
        net = self.first_layer(input)
        net = self.downsample_layers(net)
        for i in range(self.resnet_n_blocks):
            if i == 0:
                net = self._modules['resnet_layers' + str(i)](net, torch.cat(seg, 1))
            else:
                net = self._modules['resnet_layers' + str(i)](net, seg[0])
        net = self.upsample_layers(net)
        net = self.final_layer(net)
        return net


class SPADErotateGenerator(Face2FaceGenerator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9,
                            help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super(SPADErotateGenerator, self).__init__(opt)
        self.opt = opt
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        opt.norm_G = 'spectralspade' + opt.norm_G.replace('spectral', '') + '3x3'
        self.activation = nn.ReLU(False)

        mult = 1
        for i in range(opt.resnet_n_downsample):
            mult *= 2

        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            self.add_module('resnet_layers' + str(i), norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                             kernel_size=3, stride=2,
                                                             padding=1, output_padding=1)))
            self.add_module('spade_block' + str(i), SPADEResnetBlock(nc_out, nc_out, opt))

            mult = mult // 2

    def forward(self, input, seg=None):
        net = self.first_layer(input)
        net = self.downsample_layers(net)
        net = self.resnet_layers(net)
        for i in range(self.opt.resnet_n_downsample):
            net = self._modules['resnet_layers' + str(i)](net)
            net = self.activation(net)
            net = self._modules['spade_block' + str(i)]._forward(net, seg)
        net = self.final_layer(net)
        return net


class SPADErotatelightGenerator(Face2FaceGenerator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4,
                            help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9,
                            help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='spectralinstance')
        return parser

    def __init__(self, opt):
        super(SPADErotatelightGenerator, self).__init__(opt)
        self.resnet_n_blocks = opt.resnet_n_blocks
        self.opt = opt
        self.interpolate = Interpolate()
        opt.norm_G = 'spectralspade' + opt.norm_G.replace('spectral', '') + '3x3'
        mult = 1
        for i in range(opt.resnet_n_downsample):
            mult *= 2
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            self.add_module('spade_block' + str(i), SPADEResnetBlock(nc_in, nc_out, opt))
            mult = mult // 2

    def forward(self, input, seg=None):
        net = self.first_layer(input)
        net = self.downsample_layers(net)
        net = self.resnet_layers(net)
        for i in range(self.opt.resnet_n_downsample):
            net = self._modules['spade_block' + str(i)](net, seg)
            net = self.interpolate(net)
        net = self.final_layer(net)
        return net



class RotateSPADEHDGenerator(Face2FaceGenerator):
    def __init__(self, opt, localmodel):
        super(RotateSPADEHDGenerator, self).__init__(opt)
        del self.resnet_layers
        self.localmodel = localmodel
        self.resnet_n_blocks = opt.resnet_n_blocks
        mult = 1
        for i in range(opt.resnet_n_downsample):
            mult *= 2
        for i in range(opt.resnet_n_blocks):
            self.add_module('resnet_layers' + str(i), ResnetSPADEBlock(opt.ngf * mult, opt.semantic_nc))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input, seg=None):
        down_sampled_input = self.downsample(input)
        net = self.first_layer(input)
        net = self.downsample_layers(net)
        for i in range(self.resnet_n_blocks):
            net = self._modules['resnet_layers' + str(i)](net, seg)
        net = self.upsample_layers(net)
        net = self.final_layer(net)
        return net