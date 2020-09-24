from os import path

import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.transforms import ToPILImage
from torchvision.utils import save_image
from garbage_cls.dataset import GarbageDataset
from garbage_cls import data, utils
from garbage_cls.utils import draw_result, draw_heatmap, show_image, check_correct


class GarbageCls:

    def __init__(self,
                 data_root_dir,
                 checkpoints_dir,
                 ):
        print('initializing dataloaders..!')

        self.model = None

        self.labels = []
        self.num_classes = 7
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.mode = None
        self.data_root_dir = data_root_dir
        self.checkpoints_dir = checkpoints_dir

        self.device = 'cuda'
        self.transform_dict = {}

        # hyper params
        self.set_hyper_params()

        # context
        self.history = {}
        self.best_score = -1
        self.matrix = [[0 for col in range(self.num_classes)] for row in range(self.num_classes)]

        utils.touch_dir(checkpoints_dir)
        print('logging every {}, saving every {}'.format(self.log_interval, self.checkpoint_interval))
        pass

    # transform 밖에서 지정할 수 있게 하는 함수
    def set_data_transform(self, mode, transform):
        self.transform_dict[mode] = transform
        pass

    # for model
    def set_model(self, model):
        self.model = model
        pass

    # hyperparameter 역시 자주 바뀌므로 밖에서 지정할 수 있음
    def set_hyper_params(self,
                         num_epochs=100,
                         lr=0.001,
                         lr_gamma=0.5,
                         lr_milestones=None,
                         log_interval=10,
                         checkpoint_interval=1000,
                         K=5,
                         ):
        self.num_epochs = num_epochs
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.lr_milestones = lr_milestones or [40, 70]
        self.log_interval = log_interval
        self.checkpoint_interval = checkpoint_interval
        pass

    def save_checkpoint(self, name, epoch):
        torch.save({
            'model': self.model.state_dict(),
            'epoch': epoch
        }, self.get_checkpoint_path(name))
        print('{} checkpoint saved'.format(name))
        pass

    # get_check~~ : 경로 리턴, 체크포인트 가져오는 함수
    def load_checkpoint(self, name):
        checkpoint_path = self.get_checkpoint_path(name)
        if not path.exists(checkpoint_path):
            print('No {} checkpoint found.'.format(name))
            return

        data = torch.load(checkpoint_path)
        self.model.load_state_dict(data['model'])
        pass

    def get_checkpoint_path(self, name):
        return path.join(self.checkpoints_dir, name)

    def reset_context(self):
        self.history = {
            'val_loss': [],
            'val_acc': []
        }
        self.best_score = -1
        pass

    def setup(self):
        # instances
        self.criterion = nn.CrossEntropyLoss()
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = Adam(params, lr=self.lr)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.lr_milestones, gamma=self.lr_gamma)
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        pass

    # train+val
    def run(self):
        self.reset_context()
        self.setup()

        print('using {}'.format(self.device.upper()))
        print('training start')
        for epoch in range(1, self.num_epochs + 1):
            self._prepare_dataloaders()
            self._train_epoch(epoch)
            with torch.no_grad():
                self._validation_epoch(epoch)
                pass
            self.scheduler.step()  # don't forget
            pass

        draw_result(self.history)

        pass

    # test
    def test(self):
        print('using {}'.format(self.device.upper()))
        print('test start')

        self.setup()


        total = 0
        total_loss = 0
        total_correct = 0

        self.test_dataset = data.get_dataset(self.data_root_dir,
                                             'test',
                                             self._get_data_transform('test'),
                                             )
        self.test_loader = data.get_dataloader(self.test_dataset, 'test')
        print(len(self.test_loader))

        with torch.no_grad():
            for batch in self.test_loader:
                self.load_checkpoint('best')
                eval_model = self.model.eval()
                eval_model.to(self.device)

                img, label = batch
                save_img, save_label = batch

                img = img.to(self.device)
                label = label.to(self.device)
                out = eval_model(img)
                loss = self.criterion(out, label)
                _, pred = torch.max(out, dim=1)

                p, t = utils.check_correct(pred, label)

                inverse_transform = ToPILImage()

                for i in range(len(p)):
                    self.matrix[p[i]][t[i]] += 1
                    pass

                correct_list = pred.eq(label.view_as(pred))
                #print(correct_list)
                # check the wrong image
                for index in range(len(correct_list)):
                    if correct_list[index] == False:
                        torchvision.utils.save_image(save_img[index],
                                                     '/gdrive/My Drive/Capstone2020/Dataset/wrong/' + str(index) + '.png')
                        pass
                    pass

                correct = utils.get_correct_count(pred, label)

                total += img.size(0)
                total_loss += loss.item()
                total_correct += correct
                print('isCorrect / total:', correct,'/',total)
                print('nowCorrect:', total_correct)
                print()

                pass
        utils.draw_heatmap(self.matrix)
        print(total_correct, total)
        print('Accuracy: %d %%' % (100 * total_correct / total))
        pass

    def _get_data_transform(self, mode):
        if mode in self.transform_dict:
            return self.transform_dict[mode]

        return None

    def _prepare_dataloaders(self):
        self.train_dataset = data.get_dataset(self.data_root_dir,
                                              'train',
                                              self._get_data_transform('train'),
                                              )
        self.val_dataset = data.get_dataset(self.data_root_dir,
                                            'val',
                                            self._get_data_transform('val'),
                                            )

        self.train_loader = data.get_dataloader(self.train_dataset, 'train')
        self.val_loader = data.get_dataloader(self.val_dataset, 'val')

        inverse_transform = ToPILImage()

        print('current train: {} (total={})'.format(self.train_dataset.items[0], len(self.train_loader)))
        print('current val: {} (total={})'.format(self.val_dataset.items[0], len(self.val_loader)))

        print('train')

    pass

    def _train_epoch(self, epoch):
        for i, batch in enumerate(self.train_loader):
            self.model.train()
            self.optimizer.zero_grad()

            img, label = batch
            img = img.to(self.device)
            label = label.to(self.device)

            out = self.model(img)
            loss = self.criterion(out, label)
            loss.backward()
            self.optimizer.step()

            if i % self.log_interval == 0:
                current_lr = utils.get_lr(self.optimizer)
                print('E{}[{}/{}]: loss: {:.6f} lr: {}'.format(epoch,
                                                               i,
                                                               len(self.train_loader),
                                                               loss.item(),
                                                               current_lr))
            if i != 0 and i % self.checkpoint_interval == 0:
                self.save_checkpoint('latest', epoch)
        pass

    def _validation_epoch(self, epoch):
        total = 0
        total_loss = 0
        total_correct = 0

        for batch in self.val_loader:
            self.model.eval()

            img, label = batch
            img = img.to(self.device)
            label = label.to(self.device)
            out = self.model(img)
            loss = self.criterion(out, label)
            _, pred = torch.max(out, dim=1)
            correct = utils.get_correct_count(pred, label)

            total += img.size(0)
            total_loss += loss.item()
            total_correct += correct
            pass

        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / float(total)

        self.history['val_acc'].append(round(accuracy * 100, 2))
        self.history['val_loss'].append(round(avg_loss, 2))
        print(self.history)
        print('Avg. loss: {:.6f}, Accuracy: {:.2f}'.format(avg_loss, accuracy * 100))

        if accuracy > self.best_score:
            self.best_score = accuracy
            self.save_checkpoint('best', epoch)
        pass