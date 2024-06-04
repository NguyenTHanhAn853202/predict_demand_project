
# How to Run Evaluator
- Download folder from here  
https://drive.google.com/drive/folders/1a6YLGsYGF9RhgRTjwQdQROJMwAe716yn?usp=drive_link
- Place this folder as `share_participants` directory
- implement predict funtion in `Evaluator.py`
- run script
```
python Evaluator.py
```

metric will be shown
```
predicting for location: valor00001, date: 2024-04-01
predicting for location: valor00001, date: 2024-04-02
predicting for location: valor00001, date: 2024-04-03

...

predicting for location: yaoko00007, date: 2024-04-28
predicting for location: yaoko00007, date: 2024-04-29
predicting for location: yaoko00007, date: 2024-04-30
your metrics is 97.74084283562715!
```

## for real evaluation
we will add data in `2024-05-01 ~ 2024-05-31`  
and run same script for `2024-05-01 ~ 2024-05-31`

# Directory Structure 
```
.
├── Evaluator.py
├── README.md
├── requirements.txt
└── share_participants
    ├── 20240507_Competition_DEVDAY.pptx
    ├── devday2024 Q&A to share.xlsx
    ├── information session(18_00 VN time) (2024-05-09 20_04 GMT+9).mp4
    ├── specification.md.pdf
    ├── valor00001.csv
    ├── valor00001_cloud_forcast.csv
    ├── valor00001_solar_forcast.csv
    ├── valor00001_telop_forcast.csv
    ├── valor00001_temperature_forcast.csv
    ├── valor00002.csv
    ├── valor00002_cloud_forcast.csv
    ├── valor00002_solar_forcast.csv
    ├── valor00002_telop_forcast.csv
    ├── valor00002_temperature_forcast.csv
    ├── yaoko00006.csv
    ├── yaoko00006_cloud_forcast.csv
    ├── yaoko00006_solar_forcast.csv
    ├── yaoko00006_telop_forcast.csv
    ├── yaoko00006_temperature_forcast.csv
    ├── yaoko00007.csv
    ├── yaoko00007_cloud_forcast.csv
    ├── yaoko00007_solar_forcast.csv
    ├── yaoko00007_telop_forcast.csv
    └── yaoko00007_temperature_forcast.csv
```