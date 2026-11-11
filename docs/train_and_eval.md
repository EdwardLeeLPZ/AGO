# Training and Evaluation

Configs are in [`projects/configs`](../projects/configs):

| Task | Config | Epochs |
| :--- | :--- | :-: |
| Self-supervised | `ago_nuscenes_self_supervised.py` | 24 |
| Open-world pretraining | `ago_nuscenes_open_world_pretraining.py` | 24 |
| Open-world few-shot finetuning | `ago_nuscenes_open_world_finetuning.py` | 12 |

## Pretrained weights

Download the [pretrained weights](https://huggingface.co/EdwardLeeLPZ/AGO) to `checkpoints`:

```shell
for name in self_supervised open_world_pretraining open_world_finetuning; do
    wget -P checkpoints https://huggingface.co/EdwardLeeLPZ/AGO/resolve/main/ago_nuscenes_${name}.pth
done
```

## Training

```shell
bash tools/dist_train.sh projects/configs/ago_nuscenes_self_supervised.py 8 --tag self_supervised
```

Finetuning loads `checkpoints/ago_nuscenes_open_world_pretraining.pth`. Outputs are written to `work_dirs/[config]/[tag]/[stage]`.

## Evaluation

```shell
bash tools/dist_test.sh projects/configs/ago_nuscenes_self_supervised.py checkpoints/ago_nuscenes_self_supervised.pth 8 --tag self_supervised
```

Evaluating the pretraining config also reports the zero-shot results on the 15 classes of the finetuning label space, under the metric names ending with `_zs`.
