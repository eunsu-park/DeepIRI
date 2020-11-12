import argparse

class BaseOption():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--prefix', type=str, default='deepiri')
        self.parser.add_argument('--seed', type=int, default=1220)
        self.parser.add_argument('--gpu_ids', type=str, default='0, 1, 2, 3')
        self.parser.add_argument('--num_workers', type=int, default=4)

        self.parser.add_argument('--name_inp', type=str, default='iri')
        self.parser.add_argument('--name_tar', type=str, default='igs')

        self.parser.add_argument('--height', type=int, default=128)
        self.parser.add_argument('--width', type=int, default=128)

        self.parser.add_argument('--ch_inp', type=int, default=1)
        self.parser.add_argument('--ch_tar', type=int, default=1)

        self.parser.add_argument('--nb_D', type=int, default=3)
        self.parser.add_argument('--type_norm', type=str, default='none',
                                 help='[none, batch, instance]')

        self.parser.add_argument('--root_data', type=str, default='/userhome/park_e/datasets/tec')
        self.parser.add_argument('--root_save', type=str, default='/userhome/park_e/results/tec')

    def parse(self):
        return self.parser.parse_args()

class TrainOption(BaseOption):
    def __init__(self):
        super(TrainOption, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=True)
        self.parser.add_argument('--epoch_max', type=int, default=500)
        self.parser.add_argument('--display_frequency', type=int, default=100)

        self.parser.add_argument('--batch_size', type=int, default=64)
        
        self.parser.add_argument('--lr', type=float, default=0.0002)
        self.parser.add_argument('--step_size', type=int, default=100)
        self.parser.add_argument('--gamma', type=float, default=0.5)
        
        self.parser.add_argument('--beta1', type=float, default=0.5)
        self.parser.add_argument('--beta2', type=float, default=0.999)
        self.parser.add_argument('--eps', type=float, default=1e-8)

        self.parser.add_argument('--weight_FM_loss', type=float, default=10.)

class TestOption(BaseOption):
    def __init__(self):
        super(TestOption, self).__init__()

        self.parser.add_argument('--is_train', type=bool, default=False)
        self.parser.add_argument('--batch_size', type=int, default=32)
        self.parser.add_argument('--epoch_target', type=int, default=500)