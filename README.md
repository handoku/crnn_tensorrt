# crnn_tensorrt

build crnn engine from scratch with pure tensorrt api

# tensorrt version
v7.2

## build 
```bash
cd crnn
mkdir build && cd build && cmake ..
make
```
## generate engine
first, download weights file : [crnn.wts](https://drive.google.com/file/d/1g6oRrunhv1XDgF1am1e2oBJersCGRznE/view?usp=sharing) 
```bash
./crnn -s
```
## run test
```bash
./crnn -d
```