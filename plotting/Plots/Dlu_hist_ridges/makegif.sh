#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [-d delay] [-l loop] [-o output.gif] [input_directory]"
    echo "  -d delay       Set the delay between frames (in 1/100th of a second, default is 10)"
    echo "  -l loop        Set the loop count (default is 0, meaning infinite loop)"
    echo "  -o output.gif  Specify the output file name (default is animation.gif)"
    echo "  input_directory  Specify the directory containing PNG files (default is current directory)"
    exit 1
}

# Default values
DELAY=20
LOOP=0
OUTPUT="animation.gif"
INPUT_DIR="."

# Parse command line options
while getopts "d:l:o:" opt; do
    case ${opt} in
        d )
            DELAY=$OPTARG
            ;;
        l )
            LOOP=$OPTARG
            ;;
        o )
            OUTPUT=$OPTARG
            ;;
        \? )
            usage
            ;;
    esac
done
shift $((OPTIND -1))

# Check if an input directory is provided
if [ ! -z "$1" ]; then
    INPUT_DIR="$1"
fi

# Check if ImageMagick is installed
if ! command -v convert &> /dev/null
then
    echo "ImageMagick is not installed. Please install it and try again."
    exit 1
fi

# Change to the input directory
cd "$INPUT_DIR" || { echo "Directory $INPUT_DIR not found."; exit 1; }

# Convert PNG files to GIF
convert -delay "$DELAY" -loop "$LOOP" *.png "$OUTPUT"

echo "GIF created successfully: $OUTPUT"
