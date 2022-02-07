import os
import random
import torch
from torchvision.transforms import ToTensor, Compose
from torch.utils.data import Dataset
from PIL import Image
from pycocotools.coco import COCO
from typing import Any, List
from pathlib import Path


class CocoDataset(Dataset):
    def __init__(self, root: Path, annFile: Path, transforms: List = None, target_segmentations: bool = False) -> None:
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation_file=annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.target_segmentations = target_segmentations

    def __getitem__(self, index) -> Any:
        # Image ID
        img_id = self.ids[index]

        # List: get annotation id from coco
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        while len(ann_ids) == 0:
            img_id = random.sample(self.ids, 1)
            ann_ids = self.coco.getAnnIds(imgIds=img_id)

        # Dictionary: target coco_annotation file for an image
        coco_annotation = self.coco.loadAnns(ann_ids)

        # path for input image
        path = self.coco.loadImgs(img_id)[0]["file_name"]

        # open the input image
        img = Image.open(os.path.join(self.root, path))

        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes, segmentations = [], []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])
            if self.target_segmentations:
                segmentations.append(coco_annotation[i]['segmentation'])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        # Tensorise img_id
        img_id = torch.tensor([img_id])

        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]["area"])
        areas = torch.as_tensor(areas, dtype=torch.float32)

        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)
        
        retval = (img, labels, boxes)
        if self.target_segmentations:
            segmentations = torch.as_tensor(segmentations, dtype=torch.float32)
            retval += (segmentations,)

        return retval
    
    def __len__(self):
        return len(self.ids)


# In my case, just added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms += [ToTensor()]
    return Compose(custom_transforms)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))
