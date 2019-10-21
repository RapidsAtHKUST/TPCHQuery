mkdir -p build
cmake -B./build -H./q3-src -DCMAKE_INSTALL_PREFIX=. -DENABLE_LOG=OFF
make install -C build
