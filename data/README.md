# Dataset directory

Keep datasets out of Git. The default configuration expects:

```text
data/
├── images/
│   ├── image_0001.png
│   └── ...
└── labels.csv
```

`labels.csv` must include `image_id` and `diagnosis`. Labels are ICDR grades
`0` through `4`. `image_id` may omit the extension; it is added from the
configuration. If your dataset includes a patient identifier, add that column
and set `data.patient_column` in `configs/inception_v3.yaml` before preparing
splits.

No patient images, labels, model weights, or experiment outputs should be
committed to this repository.
