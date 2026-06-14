### Description
This script loads the Beijing Air Quality dataset. It reads the CSV file, displays the first 5 rows of data, and prints all column names. The dataset contains hourly air quality measurements including pollutants (PM2.5, PM10, SO2, NO2, CO, O3) along temperature, pressure, dew point, rain, wind direction, wind speed from various monitoring stations in Beijing.



### Preview:

Dataset loaded successfully.

First 5 rows
 No  year  month  day  hour  PM2.5  PM10  SO2  NO2    CO   O3  TEMP   PRES  DEWP  RAIN  wd  WSPM      station
  1  2013      3    1     0    4.0   4.0  4.0  7.0 300.0 77.0  -0.7 1023.0 -18.8   0.0 NNW   4.4 Aotizhongxin
  2  2013      3    1     1    8.0   8.0  4.0  7.0 300.0 77.0  -1.1 1023.2 -18.2   0.0   N   4.7 Aotizhongxin
  3  2013      3    1     2    7.0   7.0  5.0 10.0 300.0 73.0  -1.1 1023.5 -18.2   0.0 NNW   5.6 Aotizhongxin
  4  2013      3    1     3    6.0   6.0 11.0 11.0 300.0 72.0  -1.4 1024.5 -19.4   0.0  NW   3.1 Aotizhongxin
  5  2013      3    1     4    3.0   3.0 12.0 12.0 300.0 72.0  -2.0 1025.2 -19.5   0.0   N   2.0 Aotizhongxin
********************
Column names:
No, year, month, day, hour, PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, RAIN, wd, WSPM, station

Data types:
No           int64
year         int64
month        int64
day          int64
hour         int64
PM2.5      float64
PM10       float64
SO2        float64
NO2        float64
CO         float64
O3         float64
TEMP       float64
PRES       float64
DEWP       float64
RAIN       float64
wd             str
WSPM       float64
station        str

********************
Total rows: 420768
Total columns: 18
********************
Total Stations: 12
Name of stations: Aotizhongxin, Changping, Dingling, Dongsi, Guanyuan, Gucheng, Huairou, Nongzhanguan, Shunyi, Tiantan, Wanliu, Wanshouxigong