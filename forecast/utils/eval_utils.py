import torch
import torch.distributed as dist
import numpy as np

class multi_step_MeanIou:
    def __init__(self,
                 class_indices,
                 ignore_label: int,
                 label_str,
                 name,
                 times=1, dist=False):
        self.class_indices = class_indices
        self.num_classes = len(class_indices)
        self.ignore_label = ignore_label
        self.label_str = label_str
        self.name = name
        self.times = times
        self.dist = dist

    def reset(self) -> None:
        self.total_seen = torch.zeros(self.times, self.num_classes).cuda()
        self.total_correct = torch.zeros(self.times, self.num_classes).cuda()
        self.total_positive = torch.zeros(self.times, self.num_classes).cuda()

    def _after_step(self, outputses, targetses):

        assert outputses.shape[1] == self.times, f'{outputses.shape[1]} != {self.times}'
        assert targetses.shape[1] == self.times, f'{targetses.shape[1]} != {self.times}'
        for t in range(self.times):
            outputs = outputses[:, t, ...][targetses[:, t, ...] != self.ignore_label].cuda()
            targets = targetses[:, t, ...][targetses[:, t, ...] != self.ignore_label].cuda()
            for j, c in enumerate(self.class_indices):
                self.total_seen[t, j] += torch.sum(targets == c).item()
                self.total_correct[t, j] += torch.sum((targets == c) & (outputs == c)).item()
                self.total_positive[t, j] += torch.sum(outputs == c).item()

    def _after_epoch(self):

        if self.dist:
            dist.all_reduce(self.total_seen)
            dist.all_reduce(self.total_correct)
            dist.all_reduce(self.total_positive)
        mious = []
        for t in range(self.times):
            ious = []
            for i in range(self.num_classes):
                if self.total_seen[t, i] == 0:
                    ious.append(1)
                else:
                    cur_iou = self.total_correct[t, i] / (self.total_seen[t, i]
                                                          + self.total_positive[t, i]
                                                          - self.total_correct[t, i])
                    ious.append(cur_iou.item())
            # print(ious)
            miou = np.mean(ious)
            mious.append(miou * 100)
        return mious