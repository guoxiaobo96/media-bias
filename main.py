from webbrowser import get
from program.config import AnalysisArguments, get_config, DataArguments, MiscArgument, ModelArguments,TrainingArguments, DataAugArguments, BaselineArguments
from program.util import prepare_dirs_and_logger
from program.run_function import train_lm, label_score_predict, label_score_analysis, data_augemnt, eval_lm, label_masked_token
from program.data_collect import data_collect


def main(
        misc_args: MiscArgument,
        model_args: ModelArguments,
        data_args: DataArguments,
        aug_args: DataAugArguments,
        training_args: TrainingArguments,
        analysis_args: AnalysisArguments,
) -> None:
    if misc_args.task == 'train_lm':
        train_lm(model_args, data_args, training_args)
    elif misc_args.task == 'eval_lm':
        eval_lm(model_args, data_args, training_args)
    elif misc_args.task == 'label_masked_token':
        label_masked_token(misc_args, model_args, data_args, training_args)
    elif misc_args.task == 'label_score_predict':
        label_score_predict(misc_args, model_args, data_args,
                            training_args)
    elif misc_args.task == 'label_score_analysis':
        label_score_analysis(misc_args, model_args,
                             data_args, training_args, analysis_args,'SoA-t')
        label_score_analysis(misc_args, model_args,
                             data_args, training_args, analysis_args,'SoA-s')
        label_score_analysis(misc_args, model_args,
                             data_args, training_args, analysis_args,'MBR')
    elif misc_args.task == "data_collect":
        if aug_args.augment_type == 'original':
            data_collect(misc_args, data_args)
        else:
            data_augemnt(misc_args, data_args, aug_args)
if __name__ == '__main__':
    misc_args, model_args, data_args, aug_args, training_args, analysis_args, baseline_args = get_config()
    # misc_args.global_debug = False
    prepare_dirs_and_logger(misc_args, model_args,
                            data_args, training_args, analysis_args, baseline_args)
    main(misc_args, model_args, data_args, aug_args, 
         training_args,analysis_args)
