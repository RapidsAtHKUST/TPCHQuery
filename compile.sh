mkdir -p build
cmake -B./build -H./q3-src -DCMAKE_INSTALL_PREFIX=. -DENABLE_LOG=OFF -DENABLE_GPU=ON
make install -C build
