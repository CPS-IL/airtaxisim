#!/bin/bash

# Declare global variables
ARCH=amd64
CORES=$(grep -c ^processor /proc/cpuinfo)
ECLIPSE=4.20.0
DB_TYPE=Debug
LOCAL=1
RL_TYPE=Release

# Parse arguments
if [ "$#" -gt 0 ]; then
	if [ "$1" = "--local" ] || [ "$2" = "--local" ]; then
		LOCAL=1
	elif [ "$1" = "--system" ] || [ "$2" = "--system" ]; then
		LOCAL=0
	fi

	if [ "$1" = "--arch=amd64" ] || [ "$2" = "--arch=amd64" ]; then
		ARCH=amd64
	elif [ "$1" = "--arch=arm64" ] || [ "$2" = "--arch=arm64" ]; then
		ARCH=arm64
		LOCAL=1
	fi

	if [ "$1" != "--local" ] && [ "$2" != "--local" ] &&
		   [ "$1" = "--system" ] && [ "$2" = "--system" ] &&
		   [ "$1" = "--arch=amd64" ] && [ "$2" = "--arch=amd64" ] &&
		   [ "$1" = "--arch=arm64" ] && [ "$2" = "--arch=arm64" ]; then
		printf "Usage:\t$0\n"
		printf "\t$0 --arch=[amd64, arm64]\n"
		printf "\t$0 --local\n"
		printf "\t$0 --system\n"

		exit
	fi
fi

# Declare directory variables
PROJECT_DIR=$(pwd)/../..
SRC_DIR=$PROJECT_DIR/src
BUILD_DIR=$PROJECT_DIR/build/$ARCH
INSTALL_DIR=$PROJECT_DIR/install/$ARCH
TEMP_DIR=$PROJECT_DIR/temp

# Declare Boost variables
BOOST=boost
BOOST_DIR_BUILD=$BUILD_DIR/$BOOST
BOOST_DIR_INSTALL=$INSTALL_DIR/$BOOST
BOOST_VER_MAJ=1
BOOST_VER_MIN=65
BOOST_VER_PAT=1
BOOST_URL="https://sourceforge.net/projects/boost/files/boost/$BOOST_VER_MAJ.$BOOST_VER_MIN.$BOOST_VER_PAT/boost_${BOOST_VER_MAJ}_${BOOST_VER_MIN}_${BOOST_VER_PAT}.tar.gz/download"
BOOST_CFLAGS="-Wno-maybe-uninitialized -Wno-unused-function -Wno-unused-variable -Wno-aligned-new -Wno-parentheses -Wno-deprecated-declarations -fPIC"
BOOST_BOOTSTRAP_FLAGS="--with-libraries=system,filesystem,regex,program_options"

# Declare Eigen variables
EIGEN=eigen
EIGEN_DIR_SRC=$SRC_DIR/$EIGEN
EIGEN_DIR_BUILD=$BUILD_DIR/$EIGEN
EIGEN_DIR_INSTALL=$INSTALL_DIR/$EIGEN
EIGEN_VER_MAJ=3
EIGEN_VER_MIN=3
EIGEN_VER_PAT=4
EIGEN_URL="https://gitlab.com/libeigen/eigen/-/archive/$EIGEN_VER_MAJ.$EIGEN_VER_MIN.$EIGEN_VER_PAT/eigen-${EIGEN_VER_MAJ}.${EIGEN_VER_MIN}.${EIGEN_VER_PAT}.tar.gz"

# Declare OpenCV variables
OPENCV=opencv
OPENCV_DIR_SRC=$SRC_DIR/$OPENCV
OPENCV_DIR_BUILD=$BUILD_DIR/$OPENCV
OPENCV_DIR_INSTALL=$INSTALL_DIR/$OPENCV
OPENCV_VER_MAJ=3
OPENCV_VER_MIN=2
OPENCV_VER_PAT=0
OPENCV_URL="https://github.com/opencv/opencv/archive/${OPENCV_VER_MAJ}.${OPENCV_VER_MIN}.${OPENCV_VER_PAT}.tar.gz"
OPENCV_CFLAGS="-fPIC"
OPENCV_CMAKE_BUILD_OPTIONS="
	-DBUILD_opencv_core=ON
	-DBUILD_opencv_highgui=ON
	-DBUILD_opencv_imgproc=ON
	-DBUILD_opencv_imgcodecs=ON
	-DBUILD_opencv_calib3d=OFF
	-DBUILD_opencv_cudaarithm=OFF
	-DBUILD_opencv_cudabgsegm=OFF
	-DBUILD_opencv_cudacodec=OFF
	-DBUILD_opencv_cudafeatures2d=OFF
	-DBUILD_opencv_cudafilters=OFF
	-DBUILD_opencv_cudaimgproc=OFF
	-DBUILD_opencv_cudalegacy=OFF
	-DBUILD_opencv_cudaobjdetect=OFF
	-DBUILD_opencv_cudaoptflow=OFF
	-DBUILD_opencv_cudastereo=OFF
	-DBUILD_opencv_cudawarping=OFF
	-DBUILD_opencv_cudev=OFF
	-DBUILD_opencv_features2d=OFF
	-DBUILD_opencv_flann=OFF
	-DBUILD_opencv_java=OFF
	-DBUILD_opencv_ml=OFF
	-DBUILD_opencv_objdetect=OFF
	-DBUILD_opencv_photo=OFF
	-DBUILD_opencv_python2=OFF
	-DBUILD_opencv_python3=OFF
	-DBUILD_opencv_shape=OFF
	-DBUILD_opencv_stitching=OFF
	-DBUILD_opencv_superres=OFF
	-DBUILD_opencv_ts=OFF
	-DBUILD_opencv_video=OFF
	-DBUILD_opencv_videoio=OFF
	-DBUILD_opencv_videostab=OFF
	-DBUILD_opencv_viz=OFF
	-DBUILD_opencv_world=OFF
"
OPENCV_FILE_CMAKE_TOOLCHAIN=$OPENCV_DIR_SRC/platforms/linux/aarch64-gnu.toolchain.cmake

# Declare Depth Clustering variables
DC=depth_clustering
DC_DIR_SRC=$SRC_DIR/$DC
DC_DIR_BUILD_DB=$BUILD_DIR/$DC/$(echo ${DB_TYPE,,})
DC_DIR_BUILD_RL=$BUILD_DIR/$DC/$(echo ${RL_TYPE,,})
DC_FILE_CMAKE_TOOLCHAIN=$PROJECT_DIR/cmake/$DC/arm64.toolchain.cmake

echo "[INFO]: Setting up Depth Clustering for $ARCH architecture..."

if [ $LOCAL -eq 1 ]; then
	# Create build and temporary folders
	if [ -d "$TEMP_DIR" ]; then
		echo "[INFO]: Removing existing temporary folders..."
		rm -rf "$TEMP_DIR"
	fi
	echo "[INFO]: Creating build and temporary folders..."
	mkdir -p "$BUILD_DIR"
	mkdir -p "$TEMP_DIR"
	RETURN=$?
	if [ $RETURN -ne 0 ]; then
		echo "[ERROR]: Setup for Depth Clustering failed. Quit."
		exit $RETURN
	fi

	# Install Boost
	echo "[INFO]: Installing Boost $BOOST_VER_MAJ.$BOOST_VER_MIN.$BOOST_VER_PAT..."
	if [ ! -d "$BOOST_DIR_BUILD" ]; then
		echo "[INFO]: Downloading Boost $BOOST_VER_MAJ.$BOOST_VER_MIN.$BOOST_VER_PAT..."
		cd "$TEMP_DIR"
		curl --progress-bar -L -k "$BOOST_URL" -o "boost_${BOOST_VER_MAJ}_${BOOST_VER_MIN}_${BOOST_VER_PAT}.tar.gz"

		echo "[INFO]: Extracting Boost $BOOST_VER_MAJ.$BOOST_VER_MIN.$BOOST_VER_PAT..."
		tar -xzf "boost_${BOOST_VER_MAJ}_${BOOST_VER_MIN}_${BOOST_VER_PAT}.tar.gz"
		mv "boost_${BOOST_VER_MAJ}_${BOOST_VER_MIN}_${BOOST_VER_PAT}" "$BOOST_DIR_BUILD"
	else
		echo "[INFO]: Boost build folder exists. Skip."
	fi
	echo "[INFO]: Building Boost $BOOST_VER_MAJ.$BOOST_VER_MIN.$BOOST_VER_PAT..."
	if [ ! -d "$BOOST_DIR_INSTALL" ]; then
		cd "$BOOST_DIR_BUILD"
		./bootstrap.sh "$BOOST_BOOTSTRAP_FLAGS" --prefix="$(printf "%q" "$BOOST_DIR_INSTALL")"

		if [ "$ARCH" = "arm64" ]; then
			sed -i "/using gcc ;/c\using gcc : arm : aarch64-linux-gnu-g++ ;" project-config.jam
			./b2 cxxflags="$BOOST_CFLAGS" cflags="$BOOST_CFLAGS" link=static -j$CORES install
		else
			./b2 cxxflags="$BOOST_CFLAGS" cflags="$BOOST_CFLAGS" -j$CORES install
		fi

		RETURN=$?
		if [ $RETURN -ne 0 ]; then
			echo "[ERROR]: Setup for Depth Clustering failed. Quit."
			exit $RETURN
		fi
	else
		echo "[INFO]: Boost install folder exists. Skip."
	fi

	# Install Eigen
	echo "[INFO]: Installing Eigen $EIGEN_VER_MAJ.$EIGEN_VER_MIN.$EIGEN_VER_PAT..."
	if [ ! -d "$EIGEN_DIR_SRC" ]; then
		echo "[INFO]: Downloading Eigen $EIGEN_VER_MAJ.$EIGEN_VER_MIN.$EIGEN_VER_PAT..."
		cd "$TEMP_DIR"
		curl --progress-bar -L -k "$EIGEN_URL" -o "eigen-${EIGEN_VER_MAJ}.${EIGEN_VER_MIN}.${EIGEN_VER_PAT}.tar.gz"

		echo "[INFO]: Extracting Eigen $EIGEN_VER_MAJ.$EIGEN_VER_MIN.$EIGEN_VER_PAT..."
		tar -xzf "eigen-${EIGEN_VER_MAJ}.${EIGEN_VER_MIN}.${EIGEN_VER_PAT}.tar.gz"
		mv "eigen-${EIGEN_VER_MAJ}.${EIGEN_VER_MIN}.${EIGEN_VER_PAT}" "$EIGEN_DIR_SRC"
	else
		echo "[INFO]: Eigen source folder exists. Skip."
	fi
	echo "[INFO]: Setting up Eigen $EIGEN_VER_MAJ.$EIGEN_VER_MIN.$EIGEN_VER_PAT..."
	if [ ! -d "$EIGEN_DIR_BUILD" ]; then
		mkdir -p "$EIGEN_DIR_BUILD"
		cd "$EIGEN_DIR_BUILD"
		cmake -DEIGEN_TEST_NO_OPENGL=ON -DCMAKE_INSTALL_PREFIX="$EIGEN_DIR_INSTALL" "$EIGEN_DIR_SRC"
	else
		echo "[INFO]: Eigen build folder exists. Skip."
	fi
	echo "[INFO]: Building Eigen $EIGEN_VER_MAJ.$EIGEN_VER_MIN.$EIGEN_VER_PAT..."
	if [ ! -d "$EIGEN_DIR_INSTALL" ]; then
		cd "$EIGEN_DIR_BUILD"
		make -j$CORES install
		RETURN=$?
		if [ $RETURN -ne 0 ]; then
			echo "[ERROR]: Setup for Depth Clustering failed. Quit."
			exit $RETURN
		fi
	else
		echo "[INFO]: Eigen install folder exists. Skip."
	fi

	# Install OpenCV
	echo "[INFO]: Installing OpenCV $OPENCV_VER_MAJ.$OPENCV_VER_MIN.$OPENCV_VER_PAT..."
	if [ ! -d "$OPENCV_DIR_SRC" ]; then
		echo "[INFO]: Downloading OpenCV $OPENCV_VER_MAJ.$OPENCV_VER_MIN.$OPENCV_VER_PAT..."
		cd "$TEMP_DIR"
		curl --progress-bar -L -k "$OPENCV_URL" -o "${OPENCV_VER_MAJ}.${OPENCV_VER_MIN}.${OPENCV_VER_PAT}.tar.gz"

		echo "[INFO]: Extracting OpenCV $OPENCV_VER_MAJ.$OPENCV_VER_MIN.$OPENCV_VER_PAT..."
		tar -xzf "${OPENCV_VER_MAJ}.${OPENCV_VER_MIN}.${OPENCV_VER_PAT}.tar.gz"
		mv "opencv-${OPENCV_VER_MAJ}.${OPENCV_VER_MIN}.${OPENCV_VER_PAT}" "$OPENCV_DIR_SRC"
		sed -i "46 i find_program(CMAKE_MAKE_PROGRAM NAMES make)" "$OPENCV_DIR_SRC/platforms/linux/arm.toolchain.cmake"
	else
		echo "[INFO]: OpenCV source folder exists. Skip."
	fi
	echo "[INFO]: Setting up OpenCV $OPENCV_VER_MAJ.$OPENCV_VER_MIN.$OPENCV_VER_PAT..."
	if [ ! -d "$OPENCV_DIR_BUILD" ]; then
		mkdir -p "$OPENCV_DIR_BUILD"
		cd "$OPENCV_DIR_BUILD"

		if [ "$ARCH" = "arm64" ]; then
			cmake $OPENCV_CMAKE_BUILD_OPTIONS -DBUILD_SHARED_LIBS="OFF" -DCMAKE_INSTALL_PREFIX="$OPENCV_DIR_INSTALL" -DCMAKE_CXX_FLAGS="$OPENCV_CFLAGS" -DCMAKE_C_FLAGS="$OPENCV_CFLAGS" -DCMAKE_TOOLCHAIN_FILE="$OPENCV_FILE_CMAKE_TOOLCHAIN" -DWITH_CUDA="0" -DWITH_LAPACK="0" -DENABLE_PRECOMPILED_HEADERS=OFF "$OPENCV_DIR_SRC"
		else
			cmake $OPENCV_CMAKE_BUILD_OPTIONS -DBUILD_SHARED_LIBS="OFF" -DCMAKE_INSTALL_PREFIX="$OPENCV_DIR_INSTALL" -DCMAKE_CXX_FLAGS="$OPENCV_CFLAGS" -DCMAKE_C_FLAGS="$OPENCV_CFLAGS" -DWITH_CUDA="0" -DWITH_LAPACK="0" -DENABLE_PRECOMPILED_HEADERS=OFF "$OPENCV_DIR_SRC"
		fi
	else
		echo "[INFO]: OpenCV build folder exists. Skip."
	fi
	echo "[INFO]: Building OpenCV $OPENCV_VER_MAJ.$OPENCV_VER_MIN.$OPENCV_VER_PAT..."
	if [ ! -d "$OPENCV_DIR_INSTALL" ]; then
		cd "$OPENCV_DIR_BUILD"
		make -j$CORES install
		RETURN=$?
		if [ $RETURN -ne 0 ]; then
			echo "[ERROR]: Setup for Depth Clustering failed. Quit."
			exit $RETURN
		fi
	else
		echo "[INFO]: OpenCV install folder exists. Skip."
	fi

	# Remove temporary folder
	if [ -d "$TEMP_DIR" ]; then
		echo "[INFO]: Removing temporary folder..."
		rm -rf "$TEMP_DIR"
	fi
fi

# Create Depth Clustering projects
echo "[INFO]: Creating Depth Clustering projects..."
echo "[INFO]: Creating Depth Clustering debug project..."
if [ ! -d "$DC_DIR_BUILD_DB" ]; then
	mkdir -p "$DC_DIR_BUILD_DB"
	cd "$DC_DIR_BUILD_DB"

	if [ "$ARCH" = "arm64" ]; then
		cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=$DB_TYPE -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j$CORES -DCMAKE_ECLIPSE_VERSION=$ECLIPSE -DCMAKE_TOOLCHAIN_FILE="$DC_FILE_CMAKE_TOOLCHAIN" -DARCH=$ARCH -DLOCAL=1 "$DC_DIR_SRC"
	else
		cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=$DB_TYPE -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j$CORES -DCMAKE_ECLIPSE_VERSION=$ECLIPSE -DARCH=$ARCH -DLOCAL=$LOCAL "$DC_DIR_SRC"
	fi

	RETURN=$?
	if [ $RETURN -ne 0 ]; then
		echo "[ERROR]: Setup for Depth Clustering failed. Quit."
		exit $RETURN
	fi
else
	echo "[INFO]: Depth Clustering debug project exists. Skip."
fi
echo "[INFO]: Creating Depth Clustering release project..."
if [ ! -d "$DC_DIR_BUILD_RL" ]; then
	mkdir -p "$DC_DIR_BUILD_RL"
	cd "$DC_DIR_BUILD_RL"

	if [ "$ARCH" = "arm64" ]; then
		cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=$RL_TYPE -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j$CORES -DCMAKE_ECLIPSE_VERSION=$ECLIPSE -DCMAKE_TOOLCHAIN_FILE="$DC_FILE_CMAKE_TOOLCHAIN" -DARCH=$ARCH -DLOCAL=1 "$DC_DIR_SRC"
	else
		cmake -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_BUILD_TYPE=$RL_TYPE -DCMAKE_ECLIPSE_MAKE_ARGUMENTS=-j$CORES -DCMAKE_ECLIPSE_VERSION=$ECLIPSE -DARCH=$ARCH -DLOCAL=$LOCAL "$DC_DIR_SRC"
	fi

	RETURN=$?
	if [ $RETURN -ne 0 ]; then
		echo "[ERROR]: Setup for Depth Clustering failed. Quit."
		exit $RETURN
	fi
else
	echo "[INFO]: Depth Clustering release project exists. Skip."
fi

echo "[INFO]: Setup for Depth Clustering completed for $ARCH architecture."
