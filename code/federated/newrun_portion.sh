#bin/bash

# when --cont is set, --epochs has to be higher than the number of epochs executed when the model has been saved the last time (this information is written and retrieved from the weights file name)
# ----------------------------------------------------------------------------------------------------
python -u /home/batool/FL/federated/fed_test_all_portion.py \
--data_folder /home/batool/FL/data_half_half_size/ \
--input coord img lidar \
--id_gpu 2 \
--epochs 100 \
--lr 0.0001 \
> /home/batool/FL/federated/log_fed_test_all_gi_resume.out \
2> /home/batool/FL/federated/log_fed_test_all_gi_resume.err
