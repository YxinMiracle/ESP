import argparse


def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="ESP NER")
    parser.add_argument("--exp_name", type=str, default="ESP_experiment", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="ESP_log.log")

    parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="ESP_experiments_logs", help="Experiment id")

    parser.add_argument("--model_name", type=str, default="bert-base-cased",
                        help="model name (e.g., bert-base-cased, roberta-base)")
    parser.add_argument("--seed", type=int, default=7777, help="random seed (three seeds: 555, 666, 777)")
    parser.add_argument("--dataset_name", type=str, default="DNRTI", help="target domain")

    parser.add_argument("--ner_train_data_path", type=str, default="data/%s/train.txt")
    parser.add_argument("--ner_dev_data_path", type=str, default="data/%s/dev.txt")
    parser.add_argument("--ner_test_data_path", type=str, default="data/%s/test.txt")

    # train parameters
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epoch", type=int, default=120, help="Number of epoch")
    parser.add_argument("--shuffle", type=bool, default=True)

    parser.add_argument("--source_epoch", type=int, default=2, help="Number of epoch")

    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--early_stop", type=int, default=80,
                        help="No improvement after several epoch, we stop training")
    parser.add_argument("--num_tag", type=int, default=27, help="Number of entity in the dataset")
    parser.add_argument("--pos_tag_num", type=int, default=18, help="Number of pos in the dataset")
    parser.add_argument("--pos_embedding_dim", type=int, default=100, help="Number of entity in the dataset")
    parser.add_argument("--dropout", type=float, default=0, help="dropout rate")
    parser.add_argument("--hidden_dim", type=int, default=768, help="Hidden layer dimension")

    parser.add_argument("--save_path", type=str, default="src/model_pk/mru/dim_10/", help="embeddings file")

    # choose model
    parser.add_argument("--bilstm", default=False, action="store_true", help="use bilstm-crf structure")
    parser.add_argument("--bigru", type=bool, default=False, help="bigru Model")
    parser.add_argument("--ESP", type=bool, default=True, help="ESP Model")
    parser.add_argument("--crf", type=bool, default=False, help="crf Model")

    # biAttentionModel
    parser.add_argument("--target_embedding_dim", type=int, default=50,
                        help="conduct few-shot learning (10, 25, 40, 55, 70, 85, 100)")
    parser.add_argument("--target_sequence", default=True, action="store_true", help="use target_sequence")
    parser.add_argument("--target_type", default="LSTM", action="store_true", help="use target_sequence")
    parser.add_argument("--connect_label_background", default=True, action="store_true", help="concat label background")

    # use bigru
    parser.add_argument("--gru_hidden_dim", type=int, default=256, help="embedding dimension")

    # use BiLSTM
    parser.add_argument("--emb_dim", type=int, default=300, help="embedding dimension")
    parser.add_argument("--n_layer", type=int, default=2, help="number of layers for LSTM")
    parser.add_argument("--emb_file", type=str, default="../../glove/glove.6B.300d.txt", help="embeddings file")
    parser.add_argument("--lstm_hidden_dim", type=int, default=256, help="embedding dimension")
    parser.add_argument("--usechar", default=False, action="store_true", help="use character embeddings")
    parser.add_argument("--coach", default=False, action="store_true", help="use coach")
    parser.add_argument("--entity_enc_hidden_dim", type=int, default=300,
                        help="lstm hidden sizes for encoding entity features")
    parser.add_argument("--entity_enc_layers", type=int, default=1,
                        help="lstm encoder layers for encoding entity features")

    # few-shot
    parser.add_argument("--n_samples", type=int, default=-1,
                        help="conduct few-shot learning (10, 25, 40, 55, 70, 85, 100)")

    # seperate
    parser.add_argument("--windows", type=int, default=7, help="windows")
    params = parser.parse_args()


    return params


if __name__ == '__main__':
    params = get_params()
    print(params)
