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
```bash
./crnn -s
```
## run test
```bash
./crnn -d
```