## Code of the paper "End-to-end Learning of Temporal Logical Rules for Temporal Knowledge Graph Completion"
## Prerequisites

 * Python 3.8
 * pytorch==2.0.0


### Datasets
We use seven datasets in our experiments.

| Datasets           | Download Links (original)              |
|--------------------|----------------------------------------|
| ICEWS14            | https://github.com/liu-yushan/TLogic   |
| ICEWS18            | https://github.com/liu-yushan/TLogic   |
| ICEWS0515          | https://github.com/liu-yushan/TLogic   |
| YAGO11k            | https://github.com/seeyourmind/TKGElib |
| WIKIDATA12k        | https://github.com/seeyourmind/TKGElib |
| GDELT              | https://github.com/xyjigsaw/CENET      |


## Use examples
Running the following scripts under the ```src``` directory:

#### Training TRLM on ICEWS14

```
python main.py --data_dir ../datasets/icews14/ --exps_dir ../exps_icews14 --exp_name icews14 --batch_size 32 --use_gpu --gpu_id 0 --window 500 --length 3 --step 6 --max_epoch 10
```

#### Training TRLM on ICEWS18

```
python main.py --data_dir ../datasets/icews18/ --exps_dir ../exps_icews18 --exp_name icews18 --batch_size 32 --use_gpu --gpu_id 0 --window 500 --length 3 --max_epoch 10
```

#### Training TRLM on ICEWS0515

```
python main.py --data_dir ../datasets/icews0515/ --exps_dir ../exps_icews0515 --exp_name icews0515 --batch_size 32 --use_gpu --gpu_id 0 --window 5000 --length 3 --max_epoch 10
```

#### Training TRLM on YAGO11k

```
python main.py --data_dir ../datasets/YAGO11k --exps_dir ../exps_YAGO11k --exp_name YAGO11k --batch_size 32 --use_gpu --gpu_id 0 --window 500 --length 3 --max_epoch 10
```

#### Training TRLM on WIKIDATA12k

```
python main.py --data_dir ../datasets/WIKIDATA12k/ --exps_dir ../exps_WIKIDATA12k --exp_name WIKIDATA12k --batch_size 32 --use_gpu --gpu_id 0 --window 500 --length 3 --step 6 --max_epoch 10
```

#### Training TRLM on GDELT

```
python main.py --data_dir ../datasets/GDELT/ --exps_dir ../exps_GDELT --exp_name GDELT --batch_size 32 --use_gpu --gpu_id 0 --window 600 --length 1 --max_epoch 10
```

### Example for Rule Extraction

```
python rule_extraction.py $data_dir $exps_dir $exp_name $beam_size
```

For example, rule extraction on ICEWS14 can be:

```
python rule_extraction.py ../datasets/icews14 ../exps_icews14 icews14 1
```