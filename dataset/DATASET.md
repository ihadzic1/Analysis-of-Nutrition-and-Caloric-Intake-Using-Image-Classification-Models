# 🍎 Fruit Detection Dataset

This project uses a custom fruit detection dataset created with [Roboflow](https://roboflow.com/).

> A custom Roboflow dataset merged with a publicly available dataset from Roboflow Universe.

This project uses a modified version of the *Fruits by YOLO* dataset provided by Fruitsdetection. <br>
Citation:
```bibtex
@misc{
                            fruits-by-yolo_dataset,
                            title = { Fruits by YOLO Dataset },
                            type = { Open Source Dataset },
                            author = { Fruitsdetection },
                            howpublished = { \url{ https://universe.roboflow.com/fruitsdetection/fruits-by-yolo } },
                            url = { https://universe.roboflow.com/fruitsdetection/fruits-by-yolo },
                            journal = { Roboflow Universe },
                            publisher = { Roboflow },
                            year = { 2023 },
                            month = { feb },
                            note = { visited on 2025-06-14 },
}
```


---

## 📦 Dataset Download

You can download the dataset directly from Roboflow using the following link:

👉 [Download Dataset from Roboflow](https://app.roboflow.com/foodobjectdetection-gbrbd/foodobjectdetectiondataset)

Alternatively, you can paste the following code into your Colab notebook:

```python
!pip install roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="gbnE2mI51TkywHQXfMAH")
project = rf.workspace("foodobjectdetection-gbrbd").project("foodobjectdetectiondataset")
version = project.version(1)
dataset = version.download("yolov8")
```           

