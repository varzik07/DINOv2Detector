import os
import torch
import numpy as np
import xml.etree.ElementTree as ET 
from PIL import Image 
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF

class NEUDataset(Dataset):
    def __init__(self, root: str, transforms=None):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.imd_dir = os.path.join(root, "IMAGES")
        self.ann_dir = os.path.join(root, "ANNOTATIONS")
        self.samples = self.__load_samples()
        
        self.class_names: dict[str, int] = {}
        count = 1
        for img_path, _ in self.samples:
            cls = self._get_class_name(img_path)
            if cls not in self.class_names:
                self.class_names[cls] = count
                count += 1

    def _get_class_name(self, path: str)-> str:
        stem  = os.path.splitext(os.path.basename(path))[0]
        return "_".join(stem.split("_")[:-1])

    def __load_samples(self):
        samples = []
        for fname in sorted(os.listdir(self.imd_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg")):
                continue
            img_path = os.path.join(self.imd_dir, fname)
            xml_name = os.path.splitext(fname)[0] + ".xml"
            xml_path = os.path.join(self.ann_dir, xml_name)
            if not os.path.exists(xml_path):
                print(f"! Annotation is not founded..")
                continue
            samples.append((img_path, xml_path))
        return samples

    def _parse_xml(self, xml_path: str):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.findall("object"):
            class_name = obj.find("name").text.strip()
            if class_name not in self.class_names:
                continue
            bb = obj.find("bndbox")
            xmin, ymin, xmax, ymax = float(bb.find("xmin").text), float(bb.find("ymin").text), float(bb.find("xmax").text), float(bb.find("ymax").text)
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_names[class_name])
        return boxes, labels

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx:int):
        img_path, xml_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        boxes, labels = self._parse_xml(xml_path)

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "filename": os.path.basename(img_path)
        }

        if self.transforms:
            np_img = np.array(image)
            boxes_voc = target["boxes"].numpy().tolist()
            labels_np = target["labels"].numpy().tolist()
            try:
                transformed = self.transforms(image=np_img, bboxes=boxes_voc, labels=labels_np)
                image = transformed['image']
                
                if len(transformed['bboxes']) > 0:
                    target['boxes'] = torch.tensor(transformed['bboxes'], dtype=torch.float32)
                    target['labels'] = torch.tensor(transformed['labels'], dtype=torch.int64)
                else: 
                    target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                    target['labels'] = torch.zeros((0,), dtype=torch.int64)
            except Exception as e:
                image = TF.to_tensor(image)
                target['boxes'] = boxes_tensor
                target['labels'] = labels_tensor
        else:
            image = TF.to_tensor(image)

        return image, target


def get_dataloader(root, transformer, bs, split=[0.8, 0.1, 0.1]):
    dataset = NEUDataset(root=root, transforms=transformer)
    len_data = len(dataset)
    tr_data = int(len_data * split[0])
    val_data = int(len_data * split[1])
    ts_data = len_data - (tr_data + val_data)

    tr_ds, val_ds, ts_ds = random_split(dataset=dataset, lengths=[tr_data, val_data, ts_data])
    
    def collate_fn(batch):
        return tuple(zip(*batch))

    tr_dl = DataLoader(tr_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate_fn, num_workers=0)
    ts_dl = DataLoader(ts_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    return tr_dl, val_dl, ts_dl, dataset.class_names
