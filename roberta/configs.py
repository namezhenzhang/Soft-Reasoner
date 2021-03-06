import argparse

def get_args_parser():

    parser = argparse.ArgumentParser(description="Command line interface for Relation Extraction.")

    # Required parameters
    # parser.add_argument("--rebuild_dataset", default=0, type=int,
    #                     help="The input data dir. Should contain the data files for the task.")
    # parser.add_argument("--data_dir", default=None, type=str, required=True,
    #                     help="The input data dir. Should contain the data files for the task.")
    # parser.add_argument("--dataset", type=str, )
    # parser.add_argument("--model_type", default="albert", type=str, required=True, choices=_MODEL_CLASSES.keys(),
    #                     help="The type of the pretrained language model to use")
    # parser.add_argument("--model_name_or_path", default="albert-xxlarge-v2", type=str, required=True,
    #                     help="Path to the pre-trained model or shortcut name")
    # parser.add_argument("--output_dir_base", default="outputlogs/", type=str,
    #                     help="The output directory where the model predictions and checkpoints will be written")

    # parser.add_argument("--new_tokens", default=5, type=int, 
    #                     help="The output directory where the model predictions and checkpoints will be written")

    # parser.add_argument("--max_seq_length", default=256, type=int,
    #                     help="The maximum total input sequence length after tokenization for PET. Sequences longer "
    #                          "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    # parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
    #                     help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform in PET.")
    # parser.add_argument("--max_epochs", default=3, type=int)
                        # help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # Other optional parameters
    # parser.add_argument("--cache_dir", default="", type=str,
    #                     help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--learning_rate_for_new_token", default=1e-4, type=float,
    #                     help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0, type=float,
                        help="Linear warmup over warmup_ratio*steps.")
    parser.add_argument("--print_loss_step", default=1, type=int,
                        help="steps to print loss.")                
    # parser.add_argument('--overwrite_output_dir', action='store_true',
    #                     help="Overwrite the content of the output directory")
    # parser.add_argument('--seed', type=int, default=42,
    #                     help="random seed for initialization")

    # parser.add_argument("--temps", default="", type=str,
    #                     help="Where to store the pre-trained models downloaded from S3.")
    # parser.add_argument("--tuning", default=0, type=int,
    #                     help="0 for not tuning the prompt")

    # parser.add_argument("--kmodel", default=0, type=int,
                        # help="whether to use kmodel")
    # parser.add_argument("--mapper_path", default=None, type=str)
    # parser.add_argument("--mapper_type", default="linear", type=str)
    # parser.add_argument("--version", default="", type=str)
    # parser.add_argument("--pred_method", default="none", type=str, help=" method for get preds from logits")
    # parser.add_argument("--optimize_method", default="none", type=str, help=" method for get preds from logits")

    # parser.add_argument("--label_num", default=1, type=int, help=" num_labels_to_use")
    # # parser.add_argument("--shot", type=int, default=16)
    # parser.add_argument("--repetition", type=int, default=1)
    # parser.add_argument("--div_prior", type=int, default=-1, 
    #                     help=" whether to divide the prior of outputing the mask, \
    #                    -1 means automatically determine by zero-shot or few-shot ")
    # parser.add_argument("--label_word_file", type=str)
    # parser.add_argument("--topk", type=int, default=-1)
    # parser.add_argument("--topkratio",type=float, default=-1)
    # parser.add_argument("--autotopk",type=int, default=1)

    # parser.add_argument("--topkratioperclass", type=float, default=-1)
    # parser.add_argument("--dropwords", type=int, default=0, help="drop the verbalizer that are least possible for model prediction")
    # parser.add_argument("--template_file_name",type=str, default="template.txt")
    # parser.add_argument("--template_id", type=int, default=0)
    # parser.add_argument("--num_examples_total", type=int, default=-1)
    # parser.add_argument("--num_examples_per_label",type=int, default=-1)
    # parser.add_argument('--use_train_as_valid', type=int, default=1)
    # parser.add_argument('--run_to_save_all_logits', type=int, default=0)
    # parser.add_argument('--compute_prior', type=int, default=0)
    # parser.add_argument('--cut_off', type=float, default=0.5, help="threshold for dropping the verbalizer that are least possible for model prediction")
    # parser.add_argument('--task', type = str, default='run_single_template')
    # parser.add_argument("--only_tune_last_layer", type=int, default=0)
    args = parser.parse_args()
    # args.n_gpu = torch.cuda.device_count()

    # if args.div_prior==-1:
    #     args.div_prior= 1-args.tuning
        
    ## check arguments:
    # assert not(args.num_examples_total>0 and args.num_examples_per_label>0), "can not set these two argument at the same time"
    ##

    # if not os.path.exists(args.output_dir_base):
    #     os.mkdir(args.output_dir_base)

    # change_output_dir(args)

    # global _GLOBAL_ARGS
    # _GLOBAL_ARGS = args

    return args