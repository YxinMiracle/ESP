import argparse

RECONSTRUCTED_BASE_SENT = "The relation between \"{head_entity}\" and \"{tail_entity}\" in the context: \"{input_sent}\""


def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="YxinMiracle LLM RE")
    parser.add_argument("--bert_model_name", type=str, default="bert-base-cased",
                        help="model name (e.g., bert-base-cased, roberta-base)")
    parser.add_argument("--data_directory_name", type=str, default="data", help="data directory name")
    parser.add_argument("--save_directory_name", type=str, default="save", help="data directory name")
    parser.add_argument("--ner2idx_file_name", type=str, default="ner2idx.json", help="ner2idx file name")
    parser.add_argument("--re2idx_file_name", type=str, default="rel2idx.json", help="re2idx file name")
    parser.add_argument("--entity_relation_rule_file_name", type=str, default="entity_relation_rule.json", help="entity relation rule file name")
    parser.add_argument("--input_size", type=int, default=768, help="bert output hidden")
    parser.add_argument("--final_embedding_file_name", type=str, default="sentences_embeddings.h5", help="final embedding file name")
    parser.add_argument("--llm_model_path", type=str, default="/home/cyx/.cache/huggingface/hub/models--Qwen2-72B-Instruct", help="llm model path")
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--saved_mode_name", type=str, default="model.pt")
    parser.add_argument("--train_log_file_name", type=str, default="train.log")
    parser.add_argument("--seed", default=7777, type=int)
    parser.add_argument("--knn_num", default=10, type=int)
    parser.add_argument("--train_file_name", default="train_triples.json", type=str, help="the file name of train data")
    parser.add_argument("--gold_label_cache_file_name", default="goal_label_cache2.pt", type=str, help="gold label cache file name")
    parser.add_argument("--test_file_name", default="test_triples.json", type=str, help="the file name of test data")
    parser.add_argument("--eval_metric", default="micro", type=str, help="micro f1 or macro f1")
    parser.add_argument("--do_train", type=bool, default=True,
                        help="whether or not to train from scratch")
    parser.add_argument("--do_eval", type=bool, default=True,
                        help="whether or not to evaluate the model")
    parser.add_argument("--eval_batch_size", default=10, type=int,
                        help="number of samples in one testing batch")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="number of samples in one training batch")
    parser.add_argument("--epoch", default=100, type=int,
                        help="number of training epoch")
    parser.add_argument("--hidden_size", default=300, type=int,
                        help="number of hidden neurons in the model")
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="dropout rate for input word embedding")
    parser.add_argument("--dropconnect", default=0.1, type=float,
                        help="dropconnect rate for partition filter layer")
    parser.add_argument("--steps", default=50, type=int,
                        help="show result for every 50 steps")
    parser.add_argument("--clip", default=0.25, type=float,
                        help="grad norm clipping to avoid gradient explosion")
    parser.add_argument("--max_seq_len", default=128, type=int,
                        help="maximum length of sequence")
    parser.add_argument("--lr", default=0.0001, type=float,
                        help="initial learning rate")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="weight decaying rate")
    parser.add_argument("--linear_warmup_rate", default=0.0, type=float,
                        help="warmup at the start of training")
    params = parser.parse_args()
    return params


if __name__ == '__main__':
    params = get_params()
    print(params.bert_model_name)
