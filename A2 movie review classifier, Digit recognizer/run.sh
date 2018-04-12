#!/bin/bash

script_path="$1/$2"
question_number=$1
model=$2
if [[ $question_number = 2 && $model > 1 ]];
then 
    script_path="$script_path/libsvm/python" 
fi
python3 jsonParser.py $1 $2
cd $script_path
python3 script.py "$3" "$4" 
