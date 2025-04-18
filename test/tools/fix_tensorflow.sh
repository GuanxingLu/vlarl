# NOT USED!!!

NVIDIA_DIR=$(dirname $(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")))
for dir in $NVIDIA_DIR/*; do
   # echo $dir
    if [ -d "$dir/lib" ]; then
	echo $dir/lib
        export LD_LIBRARY_PATH="$dir/lib:$LD_LIBRARY_PATH"
    fi
done
echo "success."
