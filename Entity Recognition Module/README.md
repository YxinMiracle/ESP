**A Knowledge Extraction Framework on Cyber Threat Reports with Enhanced Security Profiles**

---

##### Requirements

- Python 3.7.16
- PyTorch 1.13.1
- Transformers 4.24.0
- Pandas 1.3.5
- Numpy 1.21.5

We use a Linux platform with A6000 GPU to train our model.

Install requirements:

```
pip install -r requirement
```

---

##### Datasets

- **DNRTI**: The DNRTI dataset is in the path `ner_data/DNRTI` and is divided into training set, validation set, and test set.There is also a vocabulary file `vocab.txt` in these three datasets, which is provided for use based on the `BILSTM`, and `BIGRU` methods.
- **MalwareTextDB**:MalwareTextDB dataset in path `ner_data/MalwareTextDB`, divided into training set, validation set and test set.There is also a vocabulary file `vocab.txt` in these three datasets, which is provided for use based on the `BILSTM`, and `BIGRU` methods.
- **Microsoft Security Bulletin**:The Microsoft Security Bulletin dataset is in `ner_data/msb` and is divided into training, validation, and test sets. This part of the dataset is not represented in the paper, but can be made available for subsequent experiments for validation.There is also a vocabulary file `vocab.txt` in these three datasets, which is provided for use based on the `BILSTM`, and `BIGRU` methods.
- <span style='color: red'>**Enterprise**: The Enterprise dataset has been withheld for corporate privacy reasons.</span>
- glove.6B.300d.txt: https://www.kaggle.com/datasets/thanakomsn/glove6b300dtxt
- glove.6B.50d.txt: https://www.kaggle.com/datasets/watts2/glove6b50dtxt

---

##### Model Training Commands

The main entry point is `main.py`. Example command:

```
python main.py \
  --exp_name ALSAP_experiment \
  --exp_id LASAP_experiments_logs \
  --logger_filename ALSAP_log.log \
  --dump_path experiments \
  --model_name bert-base-cased \
  --seed 7777 \
  --dataset_name DNRTI \
  --ner_train_data_path ner_data/%s/train.txt \
  --ner_dev_data_path ner_data/%s/dev.txt \
  --ner_test_data_path ner_data/%s/test.txt \
  --batch_size 64 \
  --epoch 120 \
  --shuffle True \
  --lr 3e-5 \
  --early_stop 20 \
  --num_tag 27 \
  --pos_tag_num 18 \
  --pos_embedding_dim 100 \
  --hidden_dim 768 \
  --save_path src/model_pk/mru/dim_10/ \
  --ALSAP True \
  --target_embedding_dim 50 \
  --target_type LSTM \
  --target_sequence True\
  --connect_label_background True\
  --n_layer 2 \
  --windows 7
```

**Hyperparameters:**

- `--exp_name`: The experiment name, which defaults to ALSAP_experiment, will be used as the root directory of the log file.
- `--exp_id`: As the id of one of the experiments under the `--exp_name` experiment, a folder named `--exp_id` will be created under the `--exp_name` folder, which will hold the log files for this `--exp_id` experiment.
- `--logger_filename`: This is used to specify the name of the log file. It will be placed under `exp_name`/`exp_id`.
- `--model_name`:  Pre-trained language model name.
- `--seed`: seed a random number.
- `--dataset_name`: The name of the dataset to select the corresponding dataset in the program.
- `--ner_train_data_path`, `--ner_dev_data_path`, `--ner_test_data_path`: The path used to find the dataset corresponding to the training, validation, and testing of the dataset.
- `--batch_size`,`--epoch`,`--shuffle`,`--lr`,`--early_stop`, `--dropout`: Base configuration for model training, which can be adapted to your own situation.
- ` --num_tag`: Number of types of data in the dataset after conversion to BIO tags.
- `--pos_tag_num`: The number of lexical types, this parameter is generally unchanged.
- `--pos_embedding_dim`: Specify the dimension of the embedding vector for lexical labels, which can be changed to suit your needs.
- `--hidden_dim`: Specify the dimensions of the word vector, which can be changed to suit your needs.
- `--save_path`:  Save the path of the best model during training.
- `--ALSAP`: Select the corresponding model type, this can contain `["ALSAP", "bigru", "bilstm", "crf"]` and so on.
- `--target_embedding_dim`: The dimension of the label embedding vector, which can be changed to suit your needs.
- `--target_type`: This is the type of context inside the proposed label, you can choose `LSTM` or the rest of the way.
- `--target_sequence`,`--target_sequence`: A control variable for the model architecture, which allows to choose whether to use labeled features or whether internal learning of labels is required.
- `--n_layer`: Controlling the number of layers in an `LSTM` or `GRU`.
- `--windows`: The variable that controls the length of the subsequence can be changed to suit your needs.

---

##### Example 

An example of using shell scripts to run the experiments in the paper:

```shell
#!/bin/bash
dims=(3 5 7 9 11 13)
for dim in "${dims[@]}"
do
	save_path="path"
    if [ ! -d "$save_path" ]; then
        mkdir -p "$save_path"
    fi
    python main.py \
    --ALSAP=True \
    --logger_filename=ALSAP_dim_300_final_windows_${dim}.log \
    --epoch=60 \
    --num_tag=27 \
    --hidden_dim="300" \
    --save_path="${save_path}" \
    --windows="${dim}" \
    --tgt_dm=huawei
done
```

