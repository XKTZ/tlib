if [ ! -p bin/$1 ]; then
  mkdir bin/$1
fi
cmake . -B export
cd export
make $1
mv $1 ../bin/$1/
cd ..
