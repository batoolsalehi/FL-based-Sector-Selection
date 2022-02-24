## FLASH: Federated Learning for Automated Selection of High-band mmWave Sectors

The present code implements FLASH framework. The FLASH dataset is available in our public repository [here](https://genesys-lab.org/multimodal-fusion-nextg-v2x-communications).

### Pre-requisites

- Python 3.8.3

- Keras 2.4.3 

- Tensorflow 2.2.0


### Cite This paper
To use this repository, please refer to our paper: 

 `@INPROCEEDINGS{flash,title = {FLASH: \underline{F}ederated \underline{L}earning for \underline{A}utomated \underline{S}election of \underline{H}igh-band mmWave Sectors}, booktitle = {{IEEE International Conference on Computer Communications (INFOCOM)}},year = "2022", author = {B. {Salehi} and J. {Gu} and D. {Roy} and K. {Chowdhury}}, month={May}, note ={[Accepted]}}`
 
 
### Run Centerlized and Local Learning:
To access the code of the FLASH architecture see "code/centralized_and_local". We use a fixed seed throughout all experiments. Run the commands below to generate the seed and global test data accordingly. Remember to change to base path to your own local machine.

        python generate_randperm.py
        python all_test.py
Finally, you can run the local and centerlized learning schemes as below:

#### Centerlized Learning:

        python main.py --data_folder path_to_data --input coord img lidar --epochs 100 --model_folder save_model_path --test_all_path path_to_global_testset_directory
#### Local Learning:

To run the local learning for each vehicle, pass the experiment_epiosdes individually as:

        python main.py --data_folder path_to_data --input coord img lidar --epochs 100 --model_folder save_model_path --test_all_path path_to_global_testset_directory --experiment_epiosdes vehicle_id
 
### Run Federated Learning framework:
To access the code of the FLASH architecture see "code/federated". A bash script is included in the repository that performs aggregation on the entire global model. Simply, run the bash script by passing the path to the data and model directory as:

        ./run_FLASH.sh path_to_data_directory path_to_save_models

To explore different aggregation policies adjust the "policy" through argparse arguments.

### Missing Information Analysis:
To access the code of the FLASH architecture see "code/Missing_info". Note that in this scenario, we are just testing. The control parameters for missing information are contorled through samples_back and prob variables in the code. 

        python main.py --data_folder path_to_data --input coord img lidar --model_folder save_model_path --test_all_path path_to_global_testset_directory 
