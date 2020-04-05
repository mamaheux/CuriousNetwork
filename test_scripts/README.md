# Test scripts
This scripts are destined to be used on Calcul Canada's computation cloud. The script_gen.py generates a bash script
for all models and all hyper-parameters that are to be tested.

## Usage
Simply call the script after having done all the required tweaks regarding models and hyper-parameters 
(see [this document](../nn/models/README.md) for more information). It will generate a bash script called start.sh
```python
python script_gen.py
```
Then call the start bash script
```bash
source start.sh
```
The latter will invoke train_corridor.sh and train_tunner.sh on its own, and start all desired trainings at once.
