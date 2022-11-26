# download file from dropbox if file does not exist
if [ ! -f Flotation_Plant_preprocessed.csv ] || [ ! -f data/Flotation_Plant_preprocessed.csv ]; then
    wget https://www.dropbox.com/s/oim8d8dnl2r3p8s/Flotation_Plant_preprocessed.csv?dl=1 -O Flotation_Plant_preprocessed.csv
fi
mkdir -p data
mv Flotation_Plant_preprocessed.csv data/Flotation_Plant_preprocessed.csv
# do preprocess
python preprocess.py