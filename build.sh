# This bash is for the lambda build-up on AWS EC2
#!/bin/bash

cd ~/LREC_Project/
virtualenv venv --python=python3
source venv/bin/activate
pip install -r ./requirement.txt

current_path=$PWD
path1="$PWD/venv/lib64/python3.6/dist-packages"
path2="$PWD/venv/lib/python3.6/dist-packages"
rm -f lambda_function.zip
zip -rj9 lambda_function.zip ./lambda/XGBoost.py 

cd $path1
zip -ur $current_path/lambda_function.zip pandas/ sklearn/

cd $path2
zip -ur $current_path/lambda_function.zip pyparsing.py six.py cycler.py pytz/ dateutil/ xgboost/

cd $current_path
echo 'Complete packaging lambda function'

aws s3api put-object --bucket lrec-datasets --key lambda_function.zip --body lambda_function.zip
echo 'Uploaded to s3 bucket'
