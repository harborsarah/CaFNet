# CaFNet: A Confidence-Driven Framework for Radar Camera Depth Estimation

Pytorch implementation of CaFNet: A Confidence-Driven Framework for Radar Camera Depth Estimation

IROS 2024

Models have been tested using Python 3.7/3.8, Pytorch 1.10.1+cu111

If you use this work, please cite our paper:

```
@INPROCEEDINGS{cafnet,
  author={Sun, Huawei and Feng, Hao and Ott, Julius and Servadei, Lorenzo and Wille, Robert},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={CaFNet: A Confidence-Driven Framework for Radar Camera Depth Estimation}, 
  year={2024},
  volume={},
  number={},
  pages={2734-2740},
  keywords={Three-dimensional displays;Radar measurements;Depth measurement;Radar;Radar imaging;Cameras;Robustness;Noise measurement;Root mean square;Surface treatment},
  doi={10.1109/IROS58592.2024.10801594}}
``` 

## Setting up dataset

Note: Run all bash scripts from the root directory.

We use the nuScenes dataset that can be downloaded [here](https://www.nuscenes.org/nuscenes#download).

Please create a folder called `dataset` and place the downloaded nuScenes dataset into it.

Generate the panoptic segmentation masks using the following:
```
python setup/gen_panoptic_seg.py
```

Then run the following bash script to generate the preprocessed dataset for training:

```
bash setup_dataset_nuscenes.sh
bash setup_dataset_nuscenes_radar.sh
```

Then run the following bash script to generate the preprocessed dataset for testing:
```
bash setup_dataset_nuscenes_test.sh
bash setup_dataset_nuscenes_radar_test.sh
```

This will generate the training dataset in a folder called `data/nuscenes_derived`

### Error in generating dataset

Note: If you meet this error "AttributeError: 'Box' object has no attribute 'box2d'" when running "bash setup_dataset_nuscenes_radar.sh", open the official nuscenes-devkit folder, go to nuscenes/utils/data_classes.py, add the following function into the Box class

```
def box2d(self, camera_intrinsic: np.ndarray, imsize: tuple=None, normalize: bool=False):

      corners_3d = self.corners()
      corners_img = view_points(points=corners_3d, view=camera_intrinsic, normalize=True)[:2, :]
      xmin = min(corners_img[0])
      xmax = max(corners_img[0])
      ymin = min(corners_img[1])
      ymax = max(corners_img[1])

      box2d = np.array([xmin, ymin, xmax, ymax])

      return box2d
```

## Training CaFNet

To train CaFNet on the nuScenes dataset, you may run

```
python main.py arguments_train_nuscenes.txt
```

## Download trained model
You can download the model weights from the link: [model](https://drive.google.com/file/d/19_XCK8ryFZsEaqVrMt4Yoc8TQHz8OVsX/view?usp=drive_link).

After downloading the model, put the file into the folder 'saved_models'. Then, it is able to evaluate the model.

## Evaluating CaFNet

To evaluate the model on the nuScenes dataset, you may run:

```
python test.py arguments_test_nuscenes.txt
```

You may replace the path dirs in the arguments files.

## Acknowledgement
Our work builds on and uses code from [radar-camera-fusion-depth](https://github.com/nesl/radar-camera-fusion-depth), [bts](https://github.com/cleinc/bts). We'd like to thank the authors for making these libraries and frameworks available.
