## FLASH: Federated Learning for Automated Selection of High-band mmWave Sectors

The present code implements FLASH framework.

### Pre-requisites

- Python 3.8.3

- Keras 2.4.3 

- tensorflow 2.2.0


### Cite This paper
To use this repository, please refer to our paper: 

 `@INPROCEEDINGS{flash,title = {FLASH: \underline{F}ederated \underline{L}earning for \underline{A}utomated \underline{S}election of \underline{H}igh-band mmWave Sectors}, booktitle = {{IEEE International Conference on Computer Communications (INFOCOM)}},year = "2022", author = {B. {Salehi} and J. {Gu} and D. {Roy} and K. {Chowdhury}}, month={May}, note ={[Accepted]}}`
 
### Run Federated Learning framework:
To access the code of the FLASH architecture see "codes/federated". A bash script is included in the repository that performs aggregation on the entire global model. Simply, run the bash script by passing the path to the data and model directory as:

        ./run_FLASH.sh path_to_data_directory path_to_save_models

To explore different aggregation policies adjust the "policy" through argparse arguments.
