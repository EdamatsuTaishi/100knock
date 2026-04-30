import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict

def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    # 長さの降順でソート
    batch.sort(key=lambda x: len(x["input_ids"]), reverse=True)

    # データの抽出
    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["label"] for item in batch]

    # パディング処理 (batch_first=Trueで [バッチサイズ, 最大長] の形状にする)
    input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    label_tensor = torch.stack(labels_list)

    return {"input_ids": input_ids_padded, "label": label_tensor}

if __name__ == "__main__":
    sample_batch = [
        {"input_ids": torch.tensor([1, 2, 3]), "label": torch.tensor(0)},
        {"input_ids": torch.tensor([4, 5]), "label": torch.tensor(1)},
        {"input_ids": torch.tensor([6, 7, 8, 9]), "label": torch.tensor(0)},
    ]

    collated_output = collate(sample_batch)
    print(collated_output)