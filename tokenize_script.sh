#!/bin/bash

# Check if the input file is provided as an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_file>"
    exit 1
fi

input_file="$1"
output_file="output.txt"

# Remove the output file if it already exists
if [ -f "$output_file" ]; then
    rm "$output_file"
fi

# Process each line in the input file
while IFS= read -r line; do
    # Tokenize the line by whitespaces
    tokens=($line)
    num_tokens=${#tokens[@]}
    
    # Keep only the last token
    last_token=${tokens[$num_tokens - 1]}
    
    # Write the last token to the output file
    echo "$last_token" >> "$output_file"
done < "$input_file"

echo "Tokenization completed. Output written to $output_file."
