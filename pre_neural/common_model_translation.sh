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



cd $WORKING_PATH/$LANGUAGES
$MOSES_FOLDER/bin/moses -f $WORKING_PATH/all-spanish/mert-work/moses.ini < $DATA_PATH/raramuri-spanish/test.true.tar > $DATA_PATH/raramuri-spanish/test.translated.all.tar 2> $DATA_PATH/raramuri-spanish/test.all.out 
$MOSES_FOLDER/bin/moses -f $WORKING_PATH/all-spanish/mert-work/moses.ini < $DATA_PATH/shipibo_konibo-spanish/test.true.shp > $DATA_PATH/shipibo_konibo-spanish/test.translated.all.shp 2> $DATA_PATH/shipibo_konibo-spanish/test.all.out 
$MOSES_FOLDER/bin/moses -f $WORKING_PATH/all-spanish/mert-work/moses.ini < $DATA_PATH/ashaninka-spanish/test.true.cni > $DATA_PATH/ashaninka-spanish/test.translated.all.cni 2> $DATA_PATH/ashaninka-spanish/test.all.out 
$MOSES_FOLDER/bin/moses -f $WORKING_PATH/all-spanish/mert-work/moses.ini < $DATA_PATH/wixarika-spanish/test.true.hch > $DATA_PATH/wixarika-spanish/test.translated.all.hch 2> $DATA_PATH/wixarika-spanish/test.all.out 
$MOSES_FOLDER/bin/moses -f $WORKING_PATH/all-spanish/mert-work/moses.ini < $DATA_PATH/aymara-spanish/test.true.aym > $DATA_PATH/aymara-spanish/test.translated.all.aym 2> $DATA_PATH/aymara-spanish/test.all.out 
$MOSES_FOLDER/bin/moses -f $WORKING_PATH/all-spanish/mert-work/moses.ini < $DATA_PATH/bribri-spanish/test.true.bzd > $DATA_PATH/bribri-spanish/test.translated.all.bzd 2> $DATA_PATH/bribri-spanish/test.all.out 
$MOSES_FOLDER/bin/moses -f $WORKING_PATH/all-spanish/mert-work/moses.ini < $DATA_PATH/guarani-spanish/test.true.gn > $DATA_PATH/guarani-spanish/test.translated.all.gn 2> $DATA_PATH/guarani-spanish/test.all.out 
$MOSES_FOLDER/bin/moses -f $WORKING_PATH/all-spanish/mert-work/moses.ini < $DATA_PATH/hñähñu-spanish/test.true.oto > $DATA_PATH/hñähñu-spanish/test.translated.all.oto 2> $DATA_PATH/hñähñu-spanish/test.all.out 
$MOSES_FOLDER/bin/moses -f $WORKING_PATH/all-spanish/mert-work/moses.ini < $DATA_PATH/nahuatl-spanish/test.true.nah > $DATA_PATH/nahuatl-spanish/test.translated.all.nah 2> $DATA_PATH/nahuatl-spanish/test.all.out 
$MOSES_FOLDER/bin/moses -f $WORKING_PATH/all-spanish/mert-work/moses.ini < $DATA_PATH/quechua-spanish/test.true.quy > $DATA_PATH/quechua-spanish/test.translated.all.quy 2> $DATA_PATH/quechua-spanish/test.all.out
