from collections import OrderedDict
from astropy import units as u
# *******************************************************
# freq[]:
#     Recording frequency For Each H2O trancition lines
#     freq[H2OJlo]=frequency[GHz]
# *******************************************************
freq=OrderedDict()
freq['[CI]609']=492160.7000/1.E3
freq['[CI]370']=809343.5000/1.E3
freq['[CII]158']=1900.53690000
freq['[NIII]57']=5231.98
freq['[NII]122']=2457000.32/1.E3
freq['[NII]205']=1461131.41/1.E3
freq['[OI]146']=2060067.53/1.E3
freq['[OI]63']=4758000.61/1.E3
freq['[OIII]88']=3406000.73/1.E3
freq['[OIII]52']=5656.4614716981
freq['[OIV]26']=11569.191448
freq['[NeV]24']=12328.259813
freq['[NeV]14']=20932.742482
freq['[NeIII]15']=19272.936722

freq['OH79']=3785.26
freq['OH119']=2510.82

freq['H2 S(0)']=10623.027462
freq['H2 S(1)']=17598.618022
freq['H2 S(2)']=24415.054809
freq['H2 S(3)']=31002.3224405
freq['H2 S(7)']=54408.7945554

freq['12CO(1-0)']=     115271.2018/1.E3
freq['12CO(2-1)']=     230538.0000/1.E3
freq['12CO(3-2)']=     345795.9899/1.E3
freq['12CO(4-3)']=     461040.7682/1.E3
freq['12CO(5-4)']=     576267.9305/1.E3
freq['12CO(6-5)']=     691473.0763/1.E3
freq['12CO(7-6)']=     806651.8060/1.E3
freq['12CO(8-7)']=     921799.7000/1.E3
freq['12CO(9-8)']=    1036912.3930/1.E3
freq['12CO(10-9)']=   1151985.4520/1.E3
freq['12CO(11-10)']=  1267014.4860/1.E3
freq['12CO(12-11)']=  1381995.1050/1.E3
freq['12CO(13-12)']=  1496922.9090/1.E3
freq['12CO(14-13)']=  1611793.5180/1.E3
freq['12CO(15-14)']=  1726602.5057/1.E3
freq['12CO(16-15)']=  1841345.5060/1.E3
freq['12CO(17-16)']=  1956018.1390/1.E3
freq['12CO(18-17)']=  2070615.9930/1.E3
freq['12CO(19-18)']=  2185134.6800/1.E3
freq['12CO(20-19)']=  2299569.8420/1.E3
freq['12CO(21-20)']=  2413917.1130/1.E3
freq['12CO(22-21)']=  2528172.0600/1.E3
freq['12CO(23-22)']=  2642330.3459/1.E3
freq['12CO(24-23)']=  2756387.5840/1.E3
freq['12CO(25-24)']=  2870339.4070/1.E3
freq['12CO(26-25)']=  2984181.4550/1.E3
freq['12CO(27-26)']=  3097909.3610/1.E3
freq['12CO(28-27)']=  3211518.7506/1.E3
freq['12CO(29-28)']=  3325005.2827/1.E3
freq['12CO(30-29)']=  3438364.6110/1.E3
freq['12CO(31-30)']=  3551592.3610/1.E3
freq['12CO(32-31)']=  3664684.1800/1.E3
freq['12CO(33-32)']=  3777635.7280/1.E3
freq['12CO(34-33)']=  3890442.7170/1.E3
freq['12CO(35-34)']=  4003100.7876/1.E3
freq['12CO(36-35)']=  4115605.5850/1.E3
freq['12CO(37-36)']=  4227952.7744/1.E3
freq['12CO(38-37)']=  4340138.1120/1.E3
freq['12CO(39-38)']=  4452157.1221/1.E3
freq['12CO(40-39)']=  4564005.6399/1.E3
freq['13CO(1-0)']=    110201.3541 /1.E3
freq['13CO(2-1)']=    220398.6765 /1.E3
freq['13CO(3-2)']=    330587.9601 /1.E3
freq['13CO(4-3)']=    440765.1668 /1.E3
freq['13CO(5-4)']=    550926.3029 /1.E3
freq['13CO(6-5)']=    661067.2801 /1.E3
freq['13CO(7-6)']=    771184.1376 /1.E3
freq['13CO(8-7)']=    881272.8339 /1.E3
freq['13CO(9-8)']=    991329.3479 /1.E3
freq['13CO(10-9)']=  1101349.6594 /1.E3
freq['13CO(11-10)']= 1211329.7493 /1.E3
freq['13CO(12-11)']= 1321265.5993 /1.E3
freq['13CO(13-12)']= 1431153.1923 /1.E3
freq['13CO(14-13)']= 1540988.5127 /1.E3
freq['13CO(15-14)']= 1650767.5458 /1.E3
freq['13CO(16-15)']= 1760486.2787 /1.E3
freq['13CO(17-16)']= 1870140.6998 /1.E3
freq['13CO(18-17)']= 1979726.7993 /1.E3
freq['13CO(19-18)']= 2089240.5692 /1.E3
freq['13CO(20-19)']= 2198678.0031 /1.E3
freq['13CO(21-20)']= 2308035.0968 /1.E3
freq['13CO(22-21)']= 2417307.8482 /1.E3
freq['13CO(23-22)']= 2526492.2570 /1.E3
freq['13CO(24-23)']= 2635584.3256 /1.E3
freq['13CO(25-24)']= 2744580.0585 /1.E3
freq['13CO(26-25)']= 2853475.4627 /1.E3
freq['13CO(27-26)']= 2962266.5478 /1.E3
freq['12C18O(1-0)']=109782.18/1.E3
freq['12C18O(2-1)']=219560.36/1.E3
freq['12C18O(3-2)']=329330.55/1.E3
freq['12C18O(4-3)']=439088.77/1.E3
freq['12C18O(5-4)']=548831.01/1.E3
freq['12C18O(6-5)']=658553.28/1.E3
freq['12C18O(7-6)']=768251.59/1.E3
freq['12C18O(8-7)']=877921.95/1.E3
freq['12C18O(9-8)']=987560.38/1.E3
freq['12C18O(10-9)']=1097162.87/1.E3
freq['12C17O(1-0)']=112.35898
freq['12C17O(2-1)']=224.71353
freq['12C17O(3-2)']=337.06061
freq['12C17O(4-3)']=449.39471
freq['12C17O(5-4)']=561.71214
freq['12C17O(6-5)']=674.00870
freq['12C17O(7-6)']=786.28017
freq['HCN(1-0)']=     88631.8470   /1.E3
freq['HCN(2-1)']=    177261.2230   /1.E3
freq['HCN(3-2)']=    265886.1800   /1.E3
freq['HCN(4-3)']=    354505.4759   /1.E3
freq['HCN(5-4)']=    443116.1554   /1.E3
freq['HCN(6-5)']=    531716.3875   /1.E3
freq['HCN(7-6)']=    620304.0952   /1.E3
freq['HCN(8-7)']=    708877.2081   /1.E3
freq['HCN(9-8)']=    797433.6638   /1.E3
freq['HCN(10-9)']=   885971.4087   /1.E3
freq['HCN(11-10)']=  974488.4000   /1.E3
freq['HCN(12-11)']= 1062982.6043   /1.E3
freq['HCN(13-12)']= 1151452.0030   /1.E3
freq['HCN(14-13)']= 1239894.5896   /1.E3
freq['HCN(15-14)']= 1328308.3723   /1.E3
freq['HCN(16-15)']= 1416691.3754   /1.E3
freq['HCN(17-16)']= 1505041.6397   /1.E3
freq['HCN(18-17)']= 1593357.2244   /1.E3
freq['HCN(19-18)']= 1681636.2074   /1.E3
freq['HCN(20-19)']= 1769876.6871   /1.E3
freq['HCN(21-20)']= 1858076.7832   /1.E3
freq['HCN(22-21)']= 1946234.6379   /1.E3
freq['HCN(23-22)']= 2034348.4171   /1.E3
freq['HCN(24-23)']= 2122416.3112   /1.E3
freq['HCN(25-24)']= 2210436.5368   /1.E3
freq['HCN(26-25)']= 2298407.3371   /1.E3
freq['HCN(27-26)']= 2386326.9838   /1.E3
freq['HCN(28-27)']= 2474193.7775   /1.E3
freq['HCN(29-28)']= 2562006.0495   /1.E3
freq['HCN(30-29)']= 2649762.1623   /1.E3
freq['HCN(31-30)']= 2737460.5112   /1.E3
freq['HCN(2-1,v2=1,I=1e)']=	177238.65/1.E3
freq['HCN(2-1,v2=1,I=1f)']=	178136.48/1.E3
freq['HCN(3-2,v2=1,I=1e)']=	265852.71/1.E3
freq['HCN(3-2,v2=1,I=1f)']=	267199.28/1.E3
freq['HCN(4-3,v2=1,I=1e)']=	354460.43/1.E3
freq['HCN(4-3,v2=1,I=1f)']=	356255.57/1.E3
freq['H13CN(1-0)']=	86339.92         /1.E3
freq['H13CN(2-1)']=	172677.84        /1.E3
freq['H13CN(3-2)']=	259011.79        /1.E3
freq['H13CN(4-3)']=	345339.76        /1.E3
freq['H13CN(5-4)']=	431659.76        /1.E3
freq['H13CN(6-5)']=	517969.81        /1.E3
freq['H13CN(7-6)']=	604267.91        /1.E3
freq['H13CN(8-7)']=	690552.07        /1.E3
freq['H13CN(9-8)']=	776820.30        /1.E3
freq['H13CN(10-9)']=	863070.62    /1.E3
freq['HCO+(1-0)']=     89188.5230    /1.E3
freq['HCO+(2-1)']=    178375.0650    /1.E3
freq['HCO+(3-2)']=    267557.6190    /1.E3
freq['HCO+(4-3)']=    356734.2880    /1.E3
freq['HCO+(5-4)']=    445902.9960    /1.E3
freq['HCO+(6-5)']=    535061.7755    /1.E3
freq['HCO+(7-6)']=    624208.6733    /1.E3
freq['HCO+(8-7)']=    713342.0900    /1.E3
freq['HCO+(9-8)']=    802458.3290    /1.E3
freq['HCO+(10-9)']=   891557.9242    /1.E3
freq['HCO+(11-10)']=  980637.4000    /1.E3
freq['HCO+(12-11)']= 1069693.8000    /1.E3
freq['HCO+(13-12)']= 1158727.6478    /1.E3
freq['HCO+(14-13)']= 1247734.8251    /1.E3
freq['HCO+(15-14)']= 1336713.8728    /1.E3
freq['HCO+(16-15)']= 1425662.7131    /1.E3
freq['HCO+(17-16)']= 1514579.2539    /1.E3
freq['HCO+(18-17)']= 1603461.3882    /1.E3
freq['HCO+(19-18)']= 1692306.9926    /1.E3
freq['HCO+(20-19)']= 1781113.9268    /1.E3
freq['HCO+(21-20)']= 1869880.0326    /1.E3
freq['HCO+(22-21)']= 1958603.1327    /1.E3
freq['HCO+(23-22)']= 2047281.0302    /1.E3
freq['HCO+(24-23)']= 2135911.5072    /1.E3
freq['HCO+(25-24)']= 2224492.3240    /1.E3
freq['HCO+(26-25)']= 2313021.2183    /1.E3
freq['HCO+(27-26)']= 2401495.9041    /1.E3
freq['HCO+(28-27)']= 2489914.0708    /1.E3
freq['HCO+(29-28)']= 2578273.3822    /1.E3
freq['HCO+(30-29)']= 2666571.4759    /1.E3
freq['HCO+(31-30)']= 2754805.9617    /1.E3
freq['H13CO+(1-0)']=	  86000.75429/1.E3
freq['H13CO+(2-1)']=	 173000.50670/1.E3
freq['H13CO+(3-2)']=	 260000.25534/1.E3
freq['H13CO+(4-3)']=	 346000.99835/1.E3
freq['H13CO+(5-4)']=	 433000.73383/1.E3
freq['H13CO+(6-5)']=	 520000.45991/1.E3
freq['H13CO+(7-6)']=	 607000.17470/1.E3
freq['H13CO+(8-7)']=	 693000.87633/1.E3
freq['H13CO+(9-8)']=	 780000.56292/1.E3
freq['H13CO+(10-9)']= 867000.23257/1.E3
freq['CS(1-0)']=     48990.9780   /1.E3
freq['CS(2-1)']=     97980.9500   /1.E3
freq['CS(3-2)']=    146969.0330   /1.E3
freq['CS(4-3)']=    195954.2260   /1.E3
freq['CS(5-4)']=    244935.6435   /1.E3
freq['CS(6-5)']=    293912.2440   /1.E3
freq['CS(7-6)']=    342883.0000   /1.E3
freq['CS(8-7)']=    391847.0300   /1.E3
freq['CS(9-8)']=    440803.3920   /1.E3
freq['CS(10-9)']=   489751.0400   /1.E3
freq['CS(11-10)']=  538688.8300   /1.E3
freq['CS(12-11)']=  587616.2402   /1.E3
freq['CS(13-12)']=  636531.8412   /1.E3
freq['CS(14-13)']=  685434.7644   /1.E3
freq['CS(15-14)']=  734323.9973   /1.E3
freq['CS(16-15)']=  783198.5190   /1.E3
freq['CS(17-16)']=  832057.2996   /1.E3
freq['CS(18-17)']=  880899.2998   /1.E3
freq['CS(19-18)']=  929723.4703   /1.E3
freq['CS(20-19)']=  978528.7509   /1.E3
freq['CS(21-20)']= 1027314.0704   /1.E3
freq['CS(22-21)']= 1076078.3456   /1.E3
freq['CS(23-22)']= 1124820.4807   /1.E3
freq['CS(24-23)']= 1173539.3671   /1.E3
freq['CS(25-24)']= 1222233.8823   /1.E3
freq['CS(26-25)']= 1270902.8897   /1.E3
freq['CS(27-26)']= 1319545.2380   /1.E3
freq['CS(28-27)']= 1368159.7602   /1.E3
freq['CS(29-28)']= 1416745.2734   /1.E3
freq['CS(30-29)']= 1465300.5781   /1.E3
freq['CS(31-30)']= 1513824.4577   /1.E3
freq['HNC(1-0)']=     90663.5930  /1.E3
freq['HNC(2-1)']=    181324.7580  /1.E3
freq['HNC(3-2)']=    271981.1420  /1.E3
freq['HNC(4-3)']=    362630.3030  /1.E3
freq['HNC(5-4)']=    453269.8531  /1.E3
freq['HNC(6-5)']=    543897.3856  /1.E3
freq['HNC(7-6)']=    634510.4973  /1.E3
freq['HNC(8-7)']=    725106.7847  /1.E3
freq['HNC(9-8)']=    815683.8445  /1.E3
freq['HNC(10-9)']=   906239.2730  /1.E3
freq['HNC(11-10)']=  996770.6669  /1.E3
freq['HNC(12-11)']= 1087275.6227  /1.E3
freq['HNC(13-12)']= 1177751.7370  /1.E3
freq['HNC(14-13)']= 1268196.6062  /1.E3
freq['HNC(15-14)']= 1358607.8268  /1.E3
freq['HNC(16-15)']= 1448982.9956  /1.E3
freq['HNC(17-16)']= 1539319.7088  /1.E3
freq['HNC(18-17)']= 1629615.5632  /1.E3
freq['HNC(19-18)']= 1719868.1552  /1.E3
freq['HNC(20-19)']= 1810075.0814  /1.E3
freq['HNC(21-20)']= 1900233.9383  /1.E3
freq['HNC(22-21)']= 1990342.3224  /1.E3
freq['HNC(23-22)']= 2080397.8304  /1.E3
freq['HNC(24-23)']= 2170398.0586  /1.E3
freq['HNC(25-24)']= 2260340.6037  /1.E3
freq['HNC(26-25)']= 2350223.0621  /1.E3
freq['HNC(27-26)']= 2440043.0305  /1.E3
freq['HNC(28-27)']= 2529798.1053  /1.E3
freq['HNC(29-28)']= 2619485.8832  /1.E3
freq['HNC(30-29)']= 2709103.9605  /1.E3
freq['HNC(31-30)']= 2798649.9339  /1.E3
freq['CN(1-0)']=	  113123.37   /1.E3
freq['CN(2-1)']=	  226287.42   /1.E3
freq['CN(3-2)']=	  339446.78   /1.E3
freq['CN(4-3)']=	  452589.69   /1.E3
freq['CN(5-4)']=	  565713.31   /1.E3
freq['CN(6-5)']=	  678813.47   /1.E3
freq['CN(7-6)']=	  791885.76   /1.E3
freq['CN(8-7)']=	  904925.64   /1.E3
freq['CN(9-8)']=	 1017928.58   /1.E3
freq['CN(10-9)']= 1130889.98      /1.E3
freq['CH+(1-0)']=  835070.9678    /1.E3
freq['CH+(2-1)']= 1669159.5100    /1.E3
freq['CH+(3-2)']= 2501284.2090    /1.E3
freq['CCH(1-0)']=	 87284.10     /1.E3
freq['CCH(2-1)']=	174549.76     /1.E3
freq['CCH(3-2)']=	261834.47     /1.E3
freq['CCH(4-3)']=	349108.50     /1.E3
freq['CCH(5-4)']=	436636.95     /1.E3
freq['CCH(6-5)']=	523948.13     /1.E3
freq['CCH(7-6)']=	611243.95     /1.E3
freq['CCH(8-7)']=	698521.91     /1.E3
freq['CCH(9-8)']=	785779.51     /1.E3
freq['CCH(10-9)']=	873014.23     /1.E3
freq['HOC+(1-0)']=     89487.4140/1.E3
freq['HOC+(2-1)']=    178972.0510/1.E3
freq['HOC+(3-2)']=    268451.0940/1.E3
freq['HOC+(4-3)']=    357921.9870/1.E3
freq['HOC+(5-4)']=    447381.7656/1.E3
freq['HOC+(6-5)']=    536827.7599/1.E3
freq['HOC+(7-6)']=    626257.1949/1.E3
freq['HOC+(8-7)']=    715667.3107/1.E3
freq['HOC+(9-8)']=    805055.3473/1.E3
freq['HOC+(10-9)']=   894418.5450/1.E3
freq['HOC+(11-10)']=  983754.1438/1.E3
freq['HOC+(12-11)']= 1073059.3838/1.E3
freq['HOC+(13-12)']= 1162331.5051/1.E3
freq['HOC+(14-13)']= 1251567.7479/1.E3
freq['HOC+(15-14)']= 1340765.3522/1.E3
freq['HOC+(16-15)']= 1429921.5582/1.E3
freq['HOC+(17-16)']= 1519033.6060/1.E3
freq['HOC+(18-17)']= 1608098.7357/1.E3
freq['HOC+(19-18)']= 1697114.1873/1.E3
freq['HOC+(20-19)']= 1786077.2011/1.E3
freq['HOC+(21-20)']= 1874985.0171/1.E3
freq['HOC+(22-21)']= 1963834.8754/1.E3
freq['HOC+(23-22)']= 2052624.0162/1.E3
freq['HOC+(24-23)']= 2141349.6796/1.E3
freq['HOC+(25-24)']= 2230009.1056/1.E3
freq['HOC+(26-25)']= 2318599.5343/1.E3
freq['HOC+(27-26)']= 2407118.2060/1.E3
freq['HOC+(28-27)']= 2495562.3607/1.E3
freq['HOC+(29-28)']= 2583929.2384/1.E3
freq['HOC+(30-29)']= 2672216.0794/1.E3
freq['HOC+(31-30)']= 2760420.1238/1.E3
freq['SiO(1-0)']=     43423.7600 /1.E3
freq['SiO(2-1)']=     86846.9600 /1.E3
freq['SiO(3-2)']=    130268.6100 /1.E3
freq['SiO(4-3)']=    173688.3100 /1.E3
freq['SiO(5-4)']=    217104.9800 /1.E3
freq['SiO(6-5)']=    260518.0200 /1.E3
freq['SiO(7-6)']=    303926.9600 /1.E3
freq['SiO(8-7)']=    347330.6310 /1.E3
freq['SiO(9-8)']=    390728.4483 /1.E3
freq['SiO(10-9)']=   434119.5521 /1.E3
freq['SiO(11-10)']=  477503.0965 /1.E3
freq['SiO(12-11)']=  520878.2039 /1.E3
freq['SiO(13-12)']=  564243.9620 /1.E3
freq['SiO(14-13)']=  607599.4207 /1.E3
freq['SiO(15-14)']=  650943.5888 /1.E3
freq['SiO(16-15)']=  694275.4309 /1.E3
freq['SiO(17-16)']=  737593.8640 /1.E3
freq['SiO(18-17)']=  780897.7550 /1.E3
freq['SiO(19-18)']=  824185.9168 /1.E3
freq['SiO(20-19)']=  867457.1055 /1.E3
freq['SiO(21-20)']=  910710.0172 /1.E3
freq['SiO(22-21)']=  953943.2851 /1.E3
freq['SiO(23-22)']=  997155.4756 /1.E3
freq['SiO(24-23)']= 1040345.0860 /1.E3
freq['SiO(25-24)']= 1083510.5408 /1.E3
freq['SiO(26-25)']= 1126650.1888 /1.E3
freq['SiO(27-26)']= 1169762.2998 /1.E3
freq['SiO(28-27)']= 1212845.0615 /1.E3
freq['SiO(29-28)']= 1255896.5762 /1.E3
freq['SiO(30-29)']= 1298914.8581 /1.E3
freq['SiO(31-30)']= 1341897.8295 /1.E3
freq['HC3N(1-0)']=      9097.0346   /1.E3
freq['HC3N(2-1)']=     18194.9360   /1.E3
freq['HC3N(3-2)']=     27292.9040   /1.E3
freq['HC3N(4-3)']=     36390.8920   /1.E3
freq['HC3N(5-4)']=     45488.8341   /1.E3
freq['HC3N(6-5)']=     54586.7291   /1.E3
freq['HC3N(7-6)']=     63686.0520   /1.E3
freq['HC3N(8-7)']=     72783.8220   /1.E3
freq['HC3N(9-8)']=     81881.4614   /1.E3
freq['HC3N(10-9)']=    90979.0230   /1.E3
freq['HC3N(11-10)']=  100076.3920   /1.E3
freq['HC3N(12-11)']=  109173.6340   /1.E3
freq['HC3N(13-12)']=  118270.7322   /1.E3
freq['HC3N(14-13)']=  127367.6660   /1.E3
freq['HC3N(15-14)']=  136464.4013   /1.E3
freq['HC3N(16-15)']=  145560.9460   /1.E3
freq['HC3N(17-16)']=  154657.2840   /1.E3
freq['HC3N(18-17)']=  163753.3890   /1.E3
freq['HC3N(19-18)']=  172849.3000   /1.E3
freq['HC3N(20-19)']=  181944.9230   /1.E3
freq['HC3N(21-20)']=  191040.2990   /1.E3
freq['HC3N(22-21)']=  200135.3920   /1.E3
freq['HC3N(23-22)']=  209230.2340   /1.E3
freq['HC3N(24-23)']=  218324.7880   /1.E3
freq['HC3N(25-24)']=  227418.9062   /1.E3
freq['HC3N(26-25)']=  236512.7768   /1.E3
freq['HC3N(27-26)']=  245606.3080   /1.E3
freq['HC3N(28-27)']=  254699.5000   /1.E3
freq['HC3N(29-28)']=  263792.3080   /1.E3
freq['HC3N(30-29)']=  272884.7343   /1.E3
freq['HC3N(31-30)']=  281976.7772   /1.E3
freq['HC3N(32-31)']=  291068.4270   /1.E3
freq['HC3N(33-32)']=  300159.6470   /1.E3
freq['HC3N(34-33)']=  309250.4300   /1.E3
freq['HC3N(35-34)']=  318340.7714   /1.E3
freq['HC3N(36-35)']=  327430.6710   /1.E3
freq['HC3N(37-36)']=  336520.0840   /1.E3
freq['HC3N(38-37)']=  345609.0100   /1.E3
freq['HC3N(39-38)']=  354697.4555   /1.E3
freq['HC3N(40-39)']=  363785.3970   /1.E3
freq['HC3N(41-40)']=  372872.8110   /1.E3
freq['HC3N(42-41)']=  381959.6694   /1.E3
freq['HC3N(43-42)']=  391045.9951   /1.E3
freq['HC3N(44-43)']=  400131.7596   /1.E3
freq['HC3N(45-44)']=  409216.9498   /1.E3
freq['HC3N(46-45)']=  418301.5529   /1.E3
freq['HC3N(47-46)']=  427385.5556   /1.E3
freq['HC3N(48-47)']=  436468.9450   /1.E3
freq['HC3N(49-48)']=  445551.7220   /1.E3
freq['HC3N(50-49)']=  454633.8316   /1.E3
freq['HC3N(51-50)']=  463715.3029   /1.E3
freq['HC3N(52-51)']=  472796.1087   /1.E3
freq['HC3N(53-52)']=  481876.2360   /1.E3
freq['HC3N(54-53)']=  490955.6719   /1.E3
freq['HC3N(55-54)']=  500034.4032   /1.E3
freq['HC3N(56-55)']=  509112.4170   /1.E3
freq['HC3N(57-56)']=  518189.7002   /1.E3
freq['HC3N(58-57)']=  527266.2399   /1.E3
freq['HC3N(59-58)']=  536342.0230   /1.E3
freq['HC3N(60-59)']=  545417.0365   /1.E3
freq['HC3N(61-60)']=  554491.2673   /1.E3
freq['HC3N(62-61)']=  563564.7026   /1.E3
freq['HC3N(63-62)']=  572637.3290   /1.E3
freq['HC3N(64-63)']=  581709.1290   /1.E3
freq['HC3N(65-64)']=  590780.0960   /1.E3
freq['HC3N(66-65)']=  599850.2270   /1.E3
freq['HC3N(67-66)']=  608919.4840   /1.E3
freq['HC3N(68-67)']=  617987.8760   /1.E3
freq['HC3N(69-68)']=  627055.3780   /1.E3
freq['HC3N(70-69)']=  636121.9780   /1.E3
freq['HC3N(71-70)']=  645187.6690   /1.E3
freq['HC3N(72-71)']=  654252.4260   /1.E3
freq['HC3N(73-72)']=  663316.2570   /1.E3
freq['HC3N(74-73)']=  672379.1310   /1.E3
freq['HC3N(75-74)']=  681441.0390   /1.E3
freq['HC3N(76-75)']=  690501.9750   /1.E3
freq['HC3N(77-76)']=  699561.9190   /1.E3
freq['HC3N(78-77)']=  708620.8500   /1.E3
freq['HC3N(79-78)']=  717678.7679   /1.E3
freq['HC3N(80-79)']=  726735.6584   /1.E3
freq['HC3N(81-80)']=  735791.5063   /1.E3
freq['HC3N(82-81)']=  744846.2987   /1.E3
freq['HC3N(83-82)']=  753899.7000   /1.E3
freq['HC3N(84-83)']=  762952.6650   /1.E3
freq['HC3N(85-84)']=  772004.2129   /1.E3
freq['HC3N(86-85)']=  781054.6534   /1.E3
freq['HC3N(87-86)']=  790104.3000   /1.E3
freq['HC3N(88-87)']=  799152.1603   /1.E3
freq['HC3N(89-88)']=  808199.2007   /1.E3
freq['HC3N(90-89)']=  817245.0818   /1.E3
freq['HC3N(91-90)']=  826289.7908   /1.E3
freq['HC3N(92-91)']=  835333.3145   /1.E3
freq['HC3N(93-92)']=  844375.6400   /1.E3
freq['HC3N(94-93)']=  853416.7545   /1.E3
freq['HC3N(95-94)']=  862456.6449   /1.E3
freq['HC3N(96-95)']=  871495.2983   /1.E3
freq['HC3N(97-96)']=  880532.7017   /1.E3
freq['HC3N(98-97)']=  889568.8423   /1.E3
freq['HC3N(99-98)']=  898603.7070   /1.E3
freq['HC3N(100-99)']= 907637.2830   /1.E3
freq['HI']=		     1420.4058      /1.E3
freq['HF(1-0)']=1232.47622
freq['HF(2-1)']=2463.42811
freq['HF(3-2)']=3691.33484
freq['HF(4-3)']=4914.68261
freq['HF(5-4)']=6131.96810
freq['HF(6-5)']=7341.70207

freq['o-H2O423312']=3807.37
freq['o-H2O523514']=1410.59
freq['o-H2O321312']=1162.93
freq['o-H2O312303']=1097.38

freq['p-H2O202111']=987.914
freq['p-H2O211202']=752.038
freq['p-H2O220211']=1228.81
# Same as above three but without the p-
freq['H2O(2_02-1_11)']=987.914
freq['H2O(2_11-2_02)']=752.038
freq['H2O(2_20-2_11)']=1228.81
freq['H2O(1_10-1_01)']=556.935995
freq['H2O(5_15-4_22)']=325.141

freq['H2O(1_10-1_01)']=556.935995
freq['o-H2O1']=556.93607
freq['o-H2O3-1']=1669.90496
freq['o-H2O5-3']=1716.76979
freq['o-H2O6-4']=1153.12682
freq['o-H2O8-5']=2640.47434
freq['o-H2O4-2']=2773.97691
freq['o-H2O10-6']=3807.25916
freq['o-H2O9-4']=4512.38479
freq['o-H2O5']=1097.36505
freq['o-H2O6']=1162.91187
freq['o-H2O13']=1410.61818
freq['p-H2O1']=1113.34306
freq['p-H2O2']=987.92670
freq['p-H2O3']=752.03323
freq['p-H2O4']=1228.78902
freq['p-H2O5']=183.31004
freq['p-H2O6']=1919.35978
freq['p-H2O8']=1602.21955
freq['p-H2O10']=916.17164
freq['p-H2O11']=325.15297
freq['p-H2O11-9']=1207.63890
freq['p-H2O6-3']=2164.13230
freq['p-H2O10-5']=4468.56976

freq['C2H(3-2)']=262.06746

## Add GHz as unit
#for line, frq in freq.items():
#     freq[line] = frq * u.GHz
