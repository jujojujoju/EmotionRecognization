# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.8

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/joju/OpenCVBlueprints/chapter_3

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/joju/OpenCVBlueprints/chapter_3/build

# Include any dependencies generated for this target.
include CMakeFiles/train.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/train.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/train.dir/flags.make

CMakeFiles/train.dir/train.cpp.o: CMakeFiles/train.dir/flags.make
CMakeFiles/train.dir/train.cpp.o: ../train.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/joju/OpenCVBlueprints/chapter_3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/train.dir/train.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/train.dir/train.cpp.o -c /Users/joju/OpenCVBlueprints/chapter_3/train.cpp

CMakeFiles/train.dir/train.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/train.dir/train.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/joju/OpenCVBlueprints/chapter_3/train.cpp > CMakeFiles/train.dir/train.cpp.i

CMakeFiles/train.dir/train.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/train.dir/train.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/joju/OpenCVBlueprints/chapter_3/train.cpp -o CMakeFiles/train.dir/train.cpp.s

CMakeFiles/train.dir/train.cpp.o.requires:

.PHONY : CMakeFiles/train.dir/train.cpp.o.requires

CMakeFiles/train.dir/train.cpp.o.provides: CMakeFiles/train.dir/train.cpp.o.requires
	$(MAKE) -f CMakeFiles/train.dir/build.make CMakeFiles/train.dir/train.cpp.o.provides.build
.PHONY : CMakeFiles/train.dir/train.cpp.o.provides

CMakeFiles/train.dir/train.cpp.o.provides.build: CMakeFiles/train.dir/train.cpp.o


# Object files for target train
train_OBJECTS = \
"CMakeFiles/train.dir/train.cpp.o"

# External object files for target train
train_EXTERNAL_OBJECTS =

train: CMakeFiles/train.dir/train.cpp.o
train: CMakeFiles/train.dir/build.make
train: /usr/local/lib/libopencv_stitching.3.2.0.dylib
train: /usr/local/lib/libopencv_superres.3.2.0.dylib
train: /usr/local/lib/libopencv_videostab.3.2.0.dylib
train: /usr/local/lib/libopencv_aruco.3.2.0.dylib
train: /usr/local/lib/libopencv_bgsegm.3.2.0.dylib
train: /usr/local/lib/libopencv_bioinspired.3.2.0.dylib
train: /usr/local/lib/libopencv_ccalib.3.2.0.dylib
train: /usr/local/lib/libopencv_dpm.3.2.0.dylib
train: /usr/local/lib/libopencv_face.3.2.0.dylib
train: /usr/local/lib/libopencv_fuzzy.3.2.0.dylib
train: /usr/local/lib/libopencv_line_descriptor.3.2.0.dylib
train: /usr/local/lib/libopencv_optflow.3.2.0.dylib
train: /usr/local/lib/libopencv_reg.3.2.0.dylib
train: /usr/local/lib/libopencv_rgbd.3.2.0.dylib
train: /usr/local/lib/libopencv_saliency.3.2.0.dylib
train: /usr/local/lib/libopencv_stereo.3.2.0.dylib
train: /usr/local/lib/libopencv_structured_light.3.2.0.dylib
train: /usr/local/lib/libopencv_surface_matching.3.2.0.dylib
train: /usr/local/lib/libopencv_tracking.3.2.0.dylib
train: /usr/local/lib/libopencv_xfeatures2d.3.2.0.dylib
train: /usr/local/lib/libopencv_ximgproc.3.2.0.dylib
train: /usr/local/lib/libopencv_xobjdetect.3.2.0.dylib
train: /usr/local/lib/libopencv_xphoto.3.2.0.dylib
train: /usr/local/lib/libopencv_shape.3.2.0.dylib
train: /usr/local/lib/libopencv_photo.3.2.0.dylib
train: /usr/local/lib/libopencv_calib3d.3.2.0.dylib
train: /usr/local/lib/libopencv_phase_unwrapping.3.2.0.dylib
train: /usr/local/lib/libopencv_video.3.2.0.dylib
train: /usr/local/lib/libopencv_datasets.3.2.0.dylib
train: /usr/local/lib/libopencv_dnn.3.2.0.dylib
train: /usr/local/lib/libopencv_plot.3.2.0.dylib
train: /usr/local/lib/libopencv_text.3.2.0.dylib
train: /usr/local/lib/libopencv_features2d.3.2.0.dylib
train: /usr/local/lib/libopencv_flann.3.2.0.dylib
train: /usr/local/lib/libopencv_highgui.3.2.0.dylib
train: /usr/local/lib/libopencv_ml.3.2.0.dylib
train: /usr/local/lib/libopencv_videoio.3.2.0.dylib
train: /usr/local/lib/libopencv_imgcodecs.3.2.0.dylib
train: /usr/local/lib/libopencv_objdetect.3.2.0.dylib
train: /usr/local/lib/libopencv_imgproc.3.2.0.dylib
train: /usr/local/lib/libopencv_core.3.2.0.dylib
train: CMakeFiles/train.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/joju/OpenCVBlueprints/chapter_3/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable train"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/train.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/train.dir/build: train

.PHONY : CMakeFiles/train.dir/build

CMakeFiles/train.dir/requires: CMakeFiles/train.dir/train.cpp.o.requires

.PHONY : CMakeFiles/train.dir/requires

CMakeFiles/train.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/train.dir/cmake_clean.cmake
.PHONY : CMakeFiles/train.dir/clean

CMakeFiles/train.dir/depend:
	cd /Users/joju/OpenCVBlueprints/chapter_3/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/joju/OpenCVBlueprints/chapter_3 /Users/joju/OpenCVBlueprints/chapter_3 /Users/joju/OpenCVBlueprints/chapter_3/build /Users/joju/OpenCVBlueprints/chapter_3/build /Users/joju/OpenCVBlueprints/chapter_3/build/CMakeFiles/train.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/train.dir/depend

