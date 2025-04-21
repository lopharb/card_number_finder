import torch


def yolo_segmentation_collate(batch):
    images = [sample['image'] for sample in batch]
    masks = [sample['masks'] for sample in batch]
    labels = [sample['labels'] for sample in batch]

    segments = []
    for i in range(len(batch)):
        for mask, cls in zip(masks[i], labels[i]):
            segments.append(torch.cat([torch.tensor([i, cls], dtype=torch.float32), mask]))

    return {
        "img": torch.stack(images),
        "segments": segments
    }
