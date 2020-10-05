# DeepFakeTorch

### Installation

You will need to install cmake first (required for dlib, which is used for face alignment).

```shell
conda create --name torchfakes
activate torchfakes
git clone https://github.com/IanSullivan/DeepFakeTorch.git
cd DeepFakeTorch
pip install requirments.txt
```

### How it works
https://www.youtube.com/watch?v=XqluthtTenI 

### Extract Faces
```shell
python facedetect.py -video_src data_src.mp4 -out_name a
python facedetect.py -video_src data_dst.mp4 -out_name b
```

---

### Train Model
```shell
python train.py -face_a_dir[location of first faces] -face_b_dir[location of seconds faces] -n_steps 100000
```

---

### Write Video
```shell
python video_writer.py -original_video [src video] -model_location saved_models/model.pt -out_name myswappedvideo -decoder [either 'a' or 'b']
```

---

## Results
Training after 10,000 steps <br>
<img src="images/swapped.gif"> <br>
<img src="images/b_to_a.gif">

## TO DO LIST (if possible)
- [ ] Smooth images on swapped video
- [ ] Add colour correction to swapped faces
