#!/bin/bash
set -e

python src/lang_chain_text_preprocessor.py

# Loop through all text files in the 2-annotated-text directory
for file in 2-annotated-text/*.txt; do
  # Get the filename without the path
  filename=$(basename "$file")
  # Get the filename without the extension
  filename_no_ext="${filename%.*}"

  # Check if the 3-output file already exists
# Check if any file with the same base name exists in the 3-output directory
if ls "3-output/${filename_no_ext}."* 1> /dev/null 2>&1; then
  echo -e "\e[90mOutput file output/${filename_no_ext} already exists. Skipping\e[0m"
  continue
fi


  # Split voice lines
  python src/split_voice_lines.py --input_file "$file"

  # Generate Zonos voice lines
  source venv_zonos/bin/activate
  python src/generate_zonos.py
  deactivate

# Generate Kokoro voice lines
  source venv_kokoro/bin/activate
  python src/generate_kokoro.py
  deactivate

  # Merge voice lines
  python src/merge_voice_lines.py

  # Clear temporary data
  rm -rf temp/*

done

