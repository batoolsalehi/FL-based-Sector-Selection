#bin/bash

# when --cont is set, --epochs has to be higher than the number of epochs executed when the model has been saved the last time (this information is written and retrieved from the weights file name)
# ----------------------------------------------------------------------------------------------------
python -u federated_learning.py \
--data_folder $1 \
--model_folder $2 \
--input coord img lidar \
--id_gpu 2 \
--epochs 100 \
--lr 0.0001 \
--aggregatin_on_all True \
> $2/log.out \
2> $2/log.err
