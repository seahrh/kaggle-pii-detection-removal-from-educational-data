kaggle-pii-detection-removal-from-educational-data
==============

```commandline
python -m mylib.train --home-dir . --conf train.ini --task ner
```

```commandline
py -3.11 -m mylib.train --home-dir . --conf train.ini --task ner
```

Kill distributed processes (https://github.com/facebookresearch/fairseq/issues/487)
```commandline
kill $(ps aux | grep "train.py" | grep -v grep | awk '{print $2}')
```
