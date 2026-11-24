# Dataset

The project uses the public **Jena Climate** dataset (2009–2016, 10-minute
observations, 14 meteorological variables).

Download the archive from:

<https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip>

Unzip it so that the repository contains:

```text
data/
  jena_climate_2009_2016.csv
```

The CSV is intentionally not committed: it is about 42 MB and publicly available.
`code/common/data.py` can also download and unpack the archive automatically on
first use; `code/lstm_common/data.py` performs the same check for the LSTM
pipeline.
