# labeling
This project is an image annotation tool. It allows a user to manually annotate image through a GUI.
It also allows the user to validate annotation through a video player, which frames are in fact jpeg files.

## Create a virtual environment
```bash
python3 -m venv env
source env/bin/activate
```

## Dependencies installation
```bash
source env/bin/activate
pip install -r requirements.txt
```

## Launch the annotation tool
```bash
source env/bin/activate
python labeling_main.py
```

#### Using the annotation tool
|Command|Key|
|-------------|:-------------:|
|Previous frame|A|
|Next frame|D|
|Annotate a section|left click|
|Unannotate a section|right click|


## Launch the annotation viewer tool
```bash
source env/bin/activate
python player_main.py
```
