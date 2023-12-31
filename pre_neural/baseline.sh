#!/bin/bash
LANGUAGES=all-spanish
SOURCE_LANGUAGE_CODE=all
TARGET_LANGUAGE_CODE=es

PROJECT_PATH=/opt/master-study/semester_2/nlp
DATA_PATH=$PROJECT_PATH/data
WORKING_PATH=$PROJECT_PATH/pre_neural/working
LM_PATH=$PROJECT_PATH/pre_neural/lm

MOSES_FOLDER=/opt/mosesdecoder
MOSES_SCRIPTS_FOLDER=$MOSES_FOLDER/scripts


mkdir -p $WORKING_PATH/$LANGUAGES
mkdir -p $LM_PATH/$LANGUAGES

cd $WORKING_PATH/$LANGUAGES
# Dataset Preparation
#   # Tokenization
$MOSES_SCRIPTS_FOLDER/tokenizer/tokenizer.perl -l $SOURCE_LANGUAGE_CODE < $DATA_PATH/$LANGUAGES/train.$SOURCE_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/train.tok.$SOURCE_LANGUAGE_CODE
$MOSES_SCRIPTS_FOLDER/tokenizer/tokenizer.perl -l $TARGET_LANGUAGE_CODE < $DATA_PATH/$LANGUAGES/train.$TARGET_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/train.tok.$TARGET_LANGUAGE_CODE
## Truecasing
$MOSES_SCRIPTS_FOLDER/recaser/train-truecaser.perl --model $DATA_PATH/$LANGUAGES/truecase-model.$SOURCE_LANGUAGE_CODE --corpus $DATA_PATH/$LANGUAGES/train.tok.$SOURCE_LANGUAGE_CODE
$MOSES_SCRIPTS_FOLDER/recaser/train-truecaser.perl --model $DATA_PATH/$LANGUAGES/truecase-model.$TARGET_LANGUAGE_CODE --corpus $DATA_PATH/$LANGUAGES/train.tok.$TARGET_LANGUAGE_CODE
$MOSES_SCRIPTS_FOLDER/recaser/truecase.perl --model $DATA_PATH/$LANGUAGES/truecase-model.$SOURCE_LANGUAGE_CODE < $DATA_PATH/$LANGUAGES/train.tok.$SOURCE_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/train.true.$SOURCE_LANGUAGE_CODE
$MOSES_SCRIPTS_FOLDER/recaser/truecase.perl --model $DATA_PATH/$LANGUAGES/truecase-model.$TARGET_LANGUAGE_CODE < $DATA_PATH/$LANGUAGES/train.tok.$TARGET_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/train.true.$TARGET_LANGUAGE_CODE

## Cleaning
$MOSES_SCRIPTS_FOLDER/training/clean-corpus-n.perl $DATA_PATH/$LANGUAGES/train.true $SOURCE_LANGUAGE_CODE $TARGET_LANGUAGE_CODE $DATA_PATH/$LANGUAGES/train.clean 1 80

# Language Model Training
$MOSES_FOLDER/bin/lmplz -o 3 < $DATA_PATH/$LANGUAGES/train.true.$TARGET_LANGUAGE_CODE > $LM_PATH/$LANGUAGES/train.arpa.$TARGET_LANGUAGE_CODE
$MOSES_FOLDER/bin/build_binary $LM_PATH/$LANGUAGES/train.arpa.$TARGET_LANGUAGE_CODE $LM_PATH/$LANGUAGES/train.blm.$TARGET_LANGUAGE_CODE

# Translation System Training
$MOSES_SCRIPTS_FOLDER/training/train-model.perl -root-dir train -corpus $DATA_PATH/$LANGUAGES/train.clean -f $SOURCE_LANGUAGE_CODE -e $TARGET_LANGUAGE_CODE -alignment grow-diag-final-and -reordering msd-bidirectional-fe -lm 0:3:$LM_PATH/$LANGUAGES/train.blm.$TARGET_LANGUAGE_CODE:8 -external-bin-dir $MOSES_FOLDER/tools -cores 8 > training.out

# Tuning
$MOSES_SCRIPTS_FOLDER/tokenizer/tokenizer.perl -l $SOURCE_LANGUAGE_CODE < $DATA_PATH/$LANGUAGES/dev.$SOURCE_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/dev.tok.$SOURCE_LANGUAGE_CODE
$MOSES_SCRIPTS_FOLDER/tokenizer/tokenizer.perl -l $TARGET_LANGUAGE_CODE < $DATA_PATH/$LANGUAGES/dev.$TARGET_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/dev.tok.$TARGET_LANGUAGE_CODE
$MOSES_SCRIPTS_FOLDER/recaser/truecase.perl --model $DATA_PATH/$LANGUAGES/truecase-model.$SOURCE_LANGUAGE_CODE < $DATA_PATH/$LANGUAGES/dev.tok.$SOURCE_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/dev.true.$SOURCE_LANGUAGE_CODE
$MOSES_SCRIPTS_FOLDER/recaser/truecase.perl --model $DATA_PATH/$LANGUAGES/truecase-model.$TARGET_LANGUAGE_CODE < $DATA_PATH/$LANGUAGES/dev.tok.$TARGET_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/dev.true.$TARGET_LANGUAGE_CODE
$MOSES_SCRIPTS_FOLDER/training/mert-moses.pl $DATA_PATH/$LANGUAGES/dev.true.$SOURCE_LANGUAGE_CODE $DATA_PATH/$LANGUAGES/dev.true.$TARGET_LANGUAGE_CODE $MOSES_FOLDER/bin/moses $WORKING_PATH/$LANGUAGES/train/model/moses.ini --mertdir $MOSES_FOLDER/bin/ > mert.out


# Testing
$MOSES_SCRIPTS_FOLDER/tokenizer/tokenizer.perl -l $SOURCE_LANGUAGE_CODE < $DATA_PATH/$LANGUAGES/test.$SOURCE_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/test.tok.$SOURCE_LANGUAGE_CODE
$MOSES_SCRIPTS_FOLDER/tokenizer/tokenizer.perl -l $TARGET_LANGUAGE_CODE < $DATA_PATH/$LANGUAGES/test.$TARGET_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/test.tok.$TARGET_LANGUAGE_CODE
$MOSES_SCRIPTS_FOLDER/recaser/truecase.perl --model $DATA_PATH/$LANGUAGES/truecase-model.$SOURCE_LANGUAGE_CODE < $DATA_PATH/$LANGUAGES/test.tok.$SOURCE_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/test.true.$SOURCE_LANGUAGE_CODE
$MOSES_SCRIPTS_FOLDER/recaser/truecase.perl --model $DATA_PATH/$LANGUAGES/truecase-model.$TARGET_LANGUAGE_CODE < $DATA_PATH/$LANGUAGES/test.tok.$TARGET_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/test.true.$TARGET_LANGUAGE_CODE

$MOSES_FOLDER/bin/moses -f $WORKING_PATH/$LANGUAGES/mert-work/moses.ini < $DATA_PATH/$LANGUAGES/test.true.$SOURCE_LANGUAGE_CODE > $DATA_PATH/$LANGUAGES/test.translated.$TARGET_LANGUAGE_CODE 2> $DATA_PATH/$LANGUAGES/test.out 
