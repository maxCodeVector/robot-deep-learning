#! /bin/sh

declare testdir

function move(){
  dirpath=$1
  files=`ls $dirpath`
  count=`ls -l $dirpath | grep "^-" | wc -l`
  cd $dirpath
  movenum=$(($count/4))
	echo "move $movenum in $count"
  for file in $files; do
    if [ $movenum -gt 0 ]; then
    	movenum=$(($movenum-1))
			echo "mv $file ../../$testdir/$dirpath/$file"
			echo ""
	   	mv $file "../../$testdir/$dirpath/$file"
		else    
			break
		fi
	done
	cd ..
}

if [ $# != 1 ]; then
	echo "must have exactly one argument: train_dir"
	exit
fi

testdir="$1_test"
echo "try create dir $testdir"
mkdir $testdir

dirpaths=`ls $1`
cd $1
for dir_p in $dirpaths; do
	if [ -d $dir_p ]; then
		echo "move file in $dir_p"
		mkdir "../$testdir/$dir_p"
		move $dir_p
	fi
done
echo "complete!"

