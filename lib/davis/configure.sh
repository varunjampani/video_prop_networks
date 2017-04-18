#! /usr/bin/env sh

#=============================================================================
#     FileName: configure.sh
#         Desc: Makefile generator
#       Author: Federico Perazzi
#        Email: federico.perazzi@disneyresearch.com
#      Version: 0.0.1
#   LastChange: 2012-03-22 21:42:25
#=============================================================================

# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

if cmake --version >/dev/null 2>&1
then
	echo
else
   echo "\nError: CMake found. Please install it. (see http://www.cmake.org/)\n"
   exit
fi

for config in Release Debug
do
	 build_type=$(echo $config | tr  '[A-Z]' '[a-z]')
   echo "------------------------------------------------------------------------"
   echo "Generating $config Makefile"
   echo "------------------------------------------------------------------------"
   mkdir -p build/$build_type
   (cd build/$build_type; cmake ../../ -DCMAKE_BUILD_TYPE:STRING=$config)
   echo
done

if [ -f build/$build_type/Makefile ] &&  [ -f build/$build_type/Makefile ]
then
	echo "##############################################################################"
	echo "Makefile successfully generated"
	echo -e "##############################################################################\n\n"

	echo -e "1. TO BUILD the source code:\n"
	echo -e " 	make -C build/{debug,release}\n"

	echo -e "2. Targets clean/cleanall have been generated for convenience\n"

	echo -e "3. Executables and libraries can be found in in ./build{debug,release}\n"

	echo -e "4. Use ccmake build/{debug,release} to visualize and/or change options.\n"

	#EOF
else

	echo -e "##############################################################################"
	echo -e "BUILDING PROCESS FAILED"
	echo -e "##############################################################################\n\n"
fi
