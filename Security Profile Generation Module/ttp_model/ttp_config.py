import pandas as pd

MODEL_NAME = "distilbert"

# 加载已经训练好的
TECHNIQUE_MODEL_PATH = "/home/cyx/ttp_recognition/save/TECHNIQUE/2024-08-10-13-40-43.pt"


TECHNIQUE = ['T1546.010', 'T1205', 'T1546', 'T1189', 'T1553.005', 'T1550', 'T1048', 'T1087.002', 'T1021.001',
             'T1598.003', 'T1200', 'T1531', 'T1491.001', 'T1132.001', 'T1055.001', 'T1498.001', 'T1555.005',
             'T1102.003', 'T1578.003', 'T1592.002', 'T1090.001', 'T1003.002', 'T1562.002', 'T1619', 'T1021.004',
             'T1134.003', 'T1029', 'T1567.002', 'T1561.001', 'T1490', 'T1011.001', 'T1518.001', 'T1210', 'T1497',
             'T1072', 'T1134.004', 'T1595.003', 'T1547.012', 'T1498.002', 'T1491', 'T1552.003', 'T1001.002',
             'T1585.001', 'T1114', 'T1098.001', 'T1542.003', 'T1622', 'T1563.001', 'T1027.005', 'T1001.001', 'T1495',
             'T1505', 'T1546.009', 'T1056.001', 'T1021.003', 'T1104', 'T1041', 'T1548.004', 'T1040', 'T1105', 'T1525',
             'T1074.001', 'T1553.006', 'T1213', 'T1547.007', 'T1589.002', 'T1078', 'T1542.005', 'T1053.007', 'T1112',
             'T1137.006', 'T1070.006', 'T1114.002', 'T1115', 'T1562.001', 'T1003.008', 'T1561', 'T1535', 'T1621',
             'T1546.012', 'T1546.014', 'T1553.002', 'T1591', 'T1578.002', 'T1012', 'T1021', 'T1053.002', 'T1195.003',
             'T1548.002', 'T1136.001', 'T1204.001', 'T1137', 'T1132', 'T1564.008', 'T1102.002', 'T1049', 'T1187',
             'T1129', 'T1574.012', 'T1070.005', 'T1573', 'T1547.004', 'T1092', 'T1555.004', 'T1037.002', 'T1596.004',
             'T1018', 'T1484.002', 'T1055.004', 'T1037', 'T1590.006', 'T1098.005', 'T1052.001', 'T1110.003',
             'T1598.002', 'T1564.010', 'T1584.003', 'T1218', 'T1211', 'T1213.003', 'T1590.003', 'T1584.001',
             'T1553.001', 'T1550.001', 'T1573.002', 'T1027.004', 'T1542.004', 'T1564.003', 'T1056.004', 'T1584.004',
             'T1027.001', 'T1647', 'T1071.001', 'T1218.003', 'T1565.001', 'T1070.004', 'T1596.003', 'T1555.001',
             'T1071.004', 'T1114.001', 'T1588.006', 'T1555.003', 'T1055.009', 'T1608.003', 'T1596.005', 'T1102',
             'T1583.006', 'T1568.003', 'T1204.002', 'T1053.005', 'T1587.004', 'T1590.001', 'T1574.009', 'T1590.002',
             'T1134.002', 'T1098', 'T1574.013', 'T1059.003', 'T1070.002', 'T1110.004', 'T1596.002', 'T1550.003',
             'T1608.005', 'T1588.002', 'T1559.003', 'T1489', 'T1574.007', 'T1559.002', 'T1098.002', 'T1030',
             'T1574.005', 'T1564.009', 'T1546.006', 'T1563.002', 'T1087.001', 'T1593.001', 'T1087.004', 'T1552.002',
             'T1568.001', 'T1047', 'T1020.001', 'T1588.001', 'T1055', 'T1176', 'T1195.001', 'T1496', 'T1055.005',
             'T1080', 'T1059.002', 'T1204', 'T1213.001', 'T1566.003', 'T1615', 'T1573.001', 'T1074', 'T1056.003',
             'T1562.008', 'T1505.001', 'T1543.003', 'T1202', 'T1595', 'T1480.001', 'T1056.002', 'T1584.005',
             'T1218.010', 'T1207', 'T1125', 'T1574.004', 'T1218.004', 'T1127', 'T1547.001', 'T1599', 'T1553', 'T1068',
             'T1547.014', 'T1069', 'T1546.005', 'T1566.002', 'T1195.002', 'T1600.001', 'T1218.013', 'T1526',
             'T1070.003', 'T1568', 'T1546.004', 'T1556.005', 'T1201', 'T1137.004', 'T1567.001', 'T1048.002',
             'T1562.003', 'T1090', 'T1203', 'T1505.005', 'T1484', 'T1059.008', 'T1059.006', 'T1609', 'T1218.012',
             'T1611', 'T1558.003', 'T1499', 'T1595.001', 'T1538', 'T1546.011', 'T1499.002', 'T1124', 'T1599.001',
             'T1608.001', 'T1027', 'T1534', 'T1110.002', 'T1574.006', 'T1003.004', 'T1053.003', 'T1001', 'T1220',
             'T1006', 'T1036.001', 'T1499.003', 'T1055.002', 'T1559', 'T1546.007', 'T1120', 'T1590', 'T1560', 'T1106',
             'T1020', 'T1578.001', 'T1594', 'T1585', 'T1595.002', 'T1055.008', 'T1558.002', 'T1499.001', 'T1055.014',
             'T1222.002', 'T1574.011', 'T1098.003', 'T1564.001', 'T1055.015', 'T1591.003', 'T1567', 'T1003',
             'T1003.005', 'T1566.001', 'T1585.002', 'T1559.001', 'T1219', 'T1114.003', 'T1588.004', 'T1132.002',
             'T1587.001', 'T1552.001', 'T1608.002', 'T1546.013', 'T1583.004', 'T1558.001', 'T1602', 'T1547.009',
             'T1606.001', 'T1027.006', 'T1003.003', 'T1588.003', 'T1543.002', 'T1102.001', 'T1547.006', 'T1037.005',
             'T1123', 'T1039', 'T1530', 'T1592.003', 'T1204.003', 'T1562.007', 'T1556', 'T1574.010', 'T1046', 'T1091',
             'T1542.001', 'T1569.002', 'T1137.002', 'T1222', 'T1596.001', 'T1195', 'T1587.002', 'T1491.002',
             'T1216.001', 'T1548.001', 'T1003.006', 'T1136', 'T1565.003', 'T1218.002', 'T1555.002', 'T1078.001',
             'T1546.001', 'T1600', 'T1557.002', 'T1090.002', 'T1614.001', 'T1558.004', 'T1036.007', 'T1505.002',
             'T1010', 'T1564.007', 'T1529', 'T1565', 'T1564.005', 'T1586', 'T1557', 'T1598', 'T1547.008', 'T1601.002',
             'T1218.008', 'T1137.001', 'T1597.002', 'T1578.004', 'T1537', 'T1586.002', 'T1547.002', 'T1036.002',
             'T1185', 'T1574', 'T1027.002', 'T1052', 'T1135', 'T1588', 'T1098.004', 'T1027.003', 'T1497.001',
             'T1586.001', 'T1016', 'T1600.002', 'T1137.005', 'T1008', 'T1136.003', 'T1003.007', 'T1583.005',
             'T1048.001', 'T1601', 'T1606', 'T1133', 'T1564.004', 'T1574.008', 'T1612', 'T1037.003', 'T1574.002',
             'T1542.002', 'T1542', 'T1048.003', 'T1059.007', 'T1218.011', 'T1583.001', 'T1071.002', 'T1070',
             'T1037.001', 'T1083', 'T1071.003', 'T1546.008', 'T1552.005', 'T1587', 'T1095', 'T1589.001', 'T1482',
             'T1003.001', 'T1497.003', 'T1557.001', 'T1021.005', 'T1036.004', 'T1602.001', 'T1557.003', 'T1528',
             'T1486', 'T1485', 'T1583', 'T1078.003', 'T1055.012', 'T1566', 'T1222.001', 'T1053.006', 'T1036.003',
             'T1016.001', 'T1055.003', 'T1221', 'T1055.013', 'T1218.001', 'T1218.014', 'T1190', 'T1553.003', 'T1571',
             'T1140', 'T1033', 'T1218.007', 'T1059.001', 'T1591.001', 'T1056', 'T1011', 'T1596', 'T1078.002',
             'T1591.004', 'T1547', 'T1561.002', 'T1082', 'T1543.004', 'T1547.010', 'T1090.004', 'T1069.002', 'T1555',
             'T1570', 'T1078.004', 'T1608', 'T1021.006', 'T1480', 'T1560.002', 'T1608.004', 'T1547.003', 'T1569',
             'T1565.002', 'T1218.005', 'T1110.001', 'T1583.002', 'T1134.001', 'T1539', 'T1550.004', 'T1087', 'T1597',
             'T1505.004', 'T1606.002', 'T1069.001', 'T1087.003', 'T1484.001', 'T1505.003', 'T1543.001', 'T1593',
             'T1614', 'T1499.004', 'T1568.002', 'T1546.003', 'T1059.005', 'T1580', 'T1553.004', 'T1552', 'T1213.002',
             'T1589', 'T1071', 'T1597.001', 'T1554', 'T1569.001', 'T1601.001', 'T1584', 'T1036', 'T1584.002', 'T1572',
             'T1556.003', 'T1036.006', 'T1591.002', 'T1199', 'T1547.015', 'T1552.006', 'T1134', 'T1074.002', 'T1216',
             'T1620', 'T1057', 'T1055.011', 'T1548.003', 'T1564', 'T1218.009', 'T1563', 'T1590.004', 'T1552.004',
             'T1005', 'T1021.002', 'T1564.002', 'T1547.013', 'T1070.001', 'T1613', 'T1588.005', 'T1025', 'T1127.001',
             'T1212', 'T1205.001', 'T1543', 'T1562', 'T1014', 'T1562.004', 'T1119', 'T1610', 'T1550.002', 'T1546.002',
             'T1111', 'T1560.001', 'T1547.005', 'T1592.004', 'T1059', 'T1498', 'T1037.004', 'T1552.007', 'T1136.002',
             'T1113', 'T1587.003', 'T1548', 'T1090.003', 'T1592', 'T1564.006', 'T1556.004', 'T1590.005', 'T1589.003',
             'T1562.010', 'T1578', 'T1562.009', 'T1562.006', 'T1598.001', 'T1592.001', 'T1110', 'T1069.003',
             'T1546.015', 'T1497.002', 'T1584.006', 'T1137.003', 'T1556.001', 'T1059.004', 'T1556.002', 'T1602.002',
             'T1593.002', 'T1583.003', 'T1574.001', 'T1134.005', 'T1518', 'T1197', 'T1036.005', 'T1558', 'T1007',
             'T1001.003', 'T1053', 'T1217', 'T1560.003']
TACTIC = ['TA0043', 'TA0042', 'TA0001', 'TA0002', 'TA0003', 'TA0004', 'TA0005', 'TA0006', 'TA0007', 'TA0008', 'TA0009',
          'TA0011', 'TA0010', 'TA0040']

TACTICS_TECHNIQUES_RELATIONSHIP_DF = {"TA0001": pd.Series(
    ['T1189', 'T1190', 'T1133', 'T1200', 'T1566', 'T1566.001', 'T1566.002', 'T1566.003', 'T1091', 'T1195', 'T1195.001',
     'T1195.002', 'T1195.003', 'T1199', 'T1078', 'T1078.001', 'T1078.002', 'T1078.003', 'T1078.004']),
    "TA0002": pd.Series(
        ['T1059', 'T1059.001', 'T1059.002', 'T1059.003', 'T1059.004', 'T1059.005',
         'T1059.006', 'T1059.007', 'T1059.008', 'T1609', 'T1610', 'T1203', 'T1559',
         'T1559.001', 'T1559.002', 'T1559.003', 'T1106', 'T1053', 'T1053.002',
         'T1053.003', 'T1053.005', 'T1053.006', 'T1053.007', 'T1129', 'T1072',
         'T1569', 'T1569.001', 'T1569.002', 'T1204', 'T1204.001', 'T1204.002',
         'T1204.003', 'T1047']),
    "TA0003": pd.Series(
        ['T1098', 'T1098.001', 'T1098.002', 'T1098.003', 'T1098.004', 'T1098.005',
         'T1197', 'T1547', 'T1547.001', 'T1547.002', 'T1547.003', 'T1547.004',
         'T1547.005', 'T1547.006', 'T1547.007', 'T1547.008', 'T1547.009', 'T1547.010',
         'T1547.012', 'T1547.013', 'T1547.014', 'T1547.015', 'T1037', 'T1037.001',
         'T1037.002', 'T1037.003', 'T1037.004', 'T1037.005', 'T1176', 'T1554',
         'T1136', 'T1136.001', 'T1136.002', 'T1136.003', 'T1543', 'T1543.001',
         'T1543.002', 'T1543.003', 'T1543.004', 'T1546', 'T1546.001', 'T1546.002',
         'T1546.003', 'T1546.004', 'T1546.005', 'T1546.006', 'T1546.007', 'T1546.008',
         'T1546.009', 'T1546.010', 'T1546.011', 'T1546.012', 'T1546.013', 'T1546.014',
         'T1546.015', 'T1133', 'T1574', 'T1574.001', 'T1574.002', 'T1574.004',
         'T1574.005', 'T1574.006', 'T1574.007', 'T1574.008', 'T1574.009', 'T1574.010',
         'T1574.011', 'T1574.012', 'T1574.013', 'T1525', 'T1556', 'T1556.001',
         'T1556.002', 'T1556.003', 'T1556.004', 'T1556.005', 'T1137', 'T1137.001',
         'T1137.002', 'T1137.003', 'T1137.004', 'T1137.005', 'T1137.006', 'T1542',
         'T1542.001', 'T1542.002', 'T1542.003', 'T1542.004', 'T1542.005', 'T1053',
         'T1053.002', 'T1053.003', 'T1053.005', 'T1053.006', 'T1053.007', 'T1505',
         'T1505.001', 'T1505.002', 'T1505.003', 'T1505.004', 'T1505.005', 'T1205',
         'T1205.001', 'T1078', 'T1078.001', 'T1078.002', 'T1078.003', 'T1078.004']),
    "TA0004": pd.Series(
        ['T1548', 'T1548.001', 'T1548.002', 'T1548.003', 'T1548.004', 'T1134',
         'T1134.001', 'T1134.002', 'T1134.003', 'T1134.004', 'T1134.005', 'T1547',
         'T1547.001', 'T1547.002', 'T1547.003', 'T1547.004', 'T1547.005', 'T1547.006',
         'T1547.007', 'T1547.008', 'T1547.009', 'T1547.010', 'T1547.012', 'T1547.013',
         'T1547.014', 'T1547.015', 'T1037', 'T1037.001', 'T1037.002', 'T1037.003',
         'T1037.004', 'T1037.005', 'T1543', 'T1543.001', 'T1543.002', 'T1543.003',
         'T1543.004', 'T1484', 'T1484.001', 'T1484.002', 'T1611', 'T1546',
         'T1546.001', 'T1546.002', 'T1546.003', 'T1546.004', 'T1546.005', 'T1546.006',
         'T1546.007', 'T1546.008', 'T1546.009', 'T1546.010', 'T1546.011', 'T1546.012',
         'T1546.013', 'T1546.014', 'T1546.015', 'T1068', 'T1574', 'T1574.001',
         'T1574.002', 'T1574.004', 'T1574.005', 'T1574.006', 'T1574.007', 'T1574.008',
         'T1574.009', 'T1574.010', 'T1574.011', 'T1574.012', 'T1574.013', 'T1055',
         'T1055.001', 'T1055.002', 'T1055.003', 'T1055.004', 'T1055.005', 'T1055.008',
         'T1055.009', 'T1055.011', 'T1055.012', 'T1055.013', 'T1055.014', 'T1055.015',
         'T1053', 'T1053.002', 'T1053.003', 'T1053.005', 'T1053.006', 'T1053.007',
         'T1078', 'T1078.001', 'T1078.002', 'T1078.003', 'T1078.004']),
    "TA0005": pd.Series(
        ['T1548', 'T1548.001', 'T1548.002', 'T1548.003', 'T1548.004', 'T1134',
         'T1134.001', 'T1134.002', 'T1134.003', 'T1134.004', 'T1134.005', 'T1197',
         'T1612', 'T1622', 'T1140', 'T1610', 'T1006', 'T1484', 'T1484.001',
         'T1484.002', 'T1480', 'T1480.001', 'T1211', 'T1222', 'T1222.001',
         'T1222.002', 'T1564', 'T1564.001', 'T1564.002', 'T1564.003', 'T1564.004',
         'T1564.005', 'T1564.006', 'T1564.007', 'T1564.008', 'T1564.009', 'T1564.010',
         'T1574', 'T1574.001', 'T1574.002', 'T1574.004', 'T1574.005', 'T1574.006',
         'T1574.007', 'T1574.008', 'T1574.009', 'T1574.010', 'T1574.011', 'T1574.012',
         'T1574.013', 'T1562', 'T1562.001', 'T1562.002', 'T1562.003', 'T1562.004',
         'T1562.006', 'T1562.007', 'T1562.008', 'T1562.009', 'T1562.010', 'T1070',
         'T1070.001', 'T1070.002', 'T1070.003', 'T1070.004', 'T1070.005', 'T1070.006',
         'T1202', 'T1036', 'T1036.001', 'T1036.002', 'T1036.003', 'T1036.004',
         'T1036.005', 'T1036.006', 'T1036.007', 'T1556', 'T1556.001', 'T1556.002',
         'T1556.003', 'T1556.004', 'T1556.005', 'T1578', 'T1578.001', 'T1578.002',
         'T1578.003', 'T1578.004', 'T1112', 'T1601', 'T1601.001', 'T1601.002',
         'T1599', 'T1599.001', 'T1027', 'T1027.001', 'T1027.002', 'T1027.003',
         'T1027.004', 'T1027.005', 'T1027.006', 'T1647', 'T1542', 'T1542.001',
         'T1542.002', 'T1542.003', 'T1542.004', 'T1542.005', 'T1055', 'T1055.001',
         'T1055.002', 'T1055.003', 'T1055.004', 'T1055.005', 'T1055.008', 'T1055.009',
         'T1055.011', 'T1055.012', 'T1055.013', 'T1055.014', 'T1055.015', 'T1620',
         'T1207', 'T1014', 'T1553', 'T1553.001', 'T1553.002', 'T1553.003',
         'T1553.004', 'T1553.005', 'T1553.006', 'T1218', 'T1218.001', 'T1218.002',
         'T1218.003', 'T1218.004', 'T1218.005', 'T1218.007', 'T1218.008', 'T1218.009',
         'T1218.010', 'T1218.011', 'T1218.012', 'T1218.013', 'T1218.014', 'T1216',
         'T1216.001', 'T1221', 'T1205', 'T1205.001', 'T1127', 'T1127.001', 'T1535',
         'T1550', 'T1550.001', 'T1550.002', 'T1550.003', 'T1550.004', 'T1078',
         'T1078.001', 'T1078.002', 'T1078.003', 'T1078.004', 'T1497', 'T1497.001',
         'T1497.002', 'T1497.003', 'T1600', 'T1600.001', 'T1600.002', 'T1220']),
    "TA0006": pd.Series(
        ['T1557', 'T1557.001', 'T1557.002', 'T1557.003', 'T1110', 'T1110.001',
         'T1110.002', 'T1110.003', 'T1110.004', 'T1555', 'T1555.001', 'T1555.002',
         'T1555.003', 'T1555.004', 'T1555.005', 'T1212', 'T1187', 'T1606',
         'T1606.001', 'T1606.002', 'T1056', 'T1056.001', 'T1056.002', 'T1056.003',
         'T1056.004', 'T1556', 'T1556.001', 'T1556.002', 'T1556.003', 'T1556.004',
         'T1556.005', 'T1111', 'T1621', 'T1040', 'T1003', 'T1003.001', 'T1003.002',
         'T1003.003', 'T1003.004', 'T1003.005', 'T1003.006', 'T1003.007', 'T1003.008',
         'T1528', 'T1558', 'T1558.001', 'T1558.002', 'T1558.003', 'T1558.004',
         'T1539', 'T1552', 'T1552.001', 'T1552.002', 'T1552.003', 'T1552.004',
         'T1552.005', 'T1552.006', 'T1552.007']),
    "TA0007": pd.Series(
        ['T1087', 'T1087.001', 'T1087.002', 'T1087.003', 'T1087.004', 'T1010',
         'T1217', 'T1580', 'T1538', 'T1526', 'T1619', 'T1613', 'T1622', 'T1482',
         'T1083', 'T1615', 'T1046', 'T1135', 'T1040', 'T1201', 'T1120', 'T1069',
         'T1069.001', 'T1069.002', 'T1069.003', 'T1057', 'T1012', 'T1018', 'T1518',
         'T1518.001', 'T1082', 'T1614', 'T1614.001', 'T1016', 'T1016.001', 'T1049',
         'T1033', 'T1007', 'T1124', 'T1497', 'T1497.001', 'T1497.002', 'T1497.003']),
    "TA0008": pd.Series(
        ['T1210', 'T1534', 'T1570', 'T1563', 'T1563.001', 'T1563.002', 'T1021',
         'T1021.001', 'T1021.002', 'T1021.003', 'T1021.004', 'T1021.005', 'T1021.006',
         'T1091', 'T1072', 'T1080', 'T1550', 'T1550.001', 'T1550.002', 'T1550.003',
         'T1550.004']),
    "TA0009": pd.Series(
        ['T1557', 'T1557.001', 'T1557.002', 'T1557.003', 'T1560', 'T1560.001',
         'T1560.002', 'T1560.003', 'T1123', 'T1119', 'T1185', 'T1115', 'T1530',
         'T1602', 'T1602.001', 'T1602.002', 'T1213', 'T1213.001', 'T1213.002',
         'T1213.003', 'T1005', 'T1039', 'T1025', 'T1074', 'T1074.001', 'T1074.002',
         'T1114', 'T1114.001', 'T1114.002', 'T1114.003', 'T1056', 'T1056.001',
         'T1056.002', 'T1056.003', 'T1056.004', 'T1113', 'T1125']),
    "TA0010": pd.Series(
        ['T1020', 'T1020.001', 'T1030', 'T1048', 'T1048.001', 'T1048.002',
         'T1048.003', 'T1041', 'T1011', 'T1011.001', 'T1052', 'T1052.001', 'T1567',
         'T1567.001', 'T1567.002', 'T1029', 'T1537']),
    "TA0011": pd.Series(
        ['T1071', 'T1071.001', 'T1071.002', 'T1071.003', 'T1071.004', 'T1092',
         'T1132', 'T1132.001', 'T1132.002', 'T1001', 'T1001.001', 'T1001.002',
         'T1001.003', 'T1568', 'T1568.001', 'T1568.002', 'T1568.003', 'T1573',
         'T1573.001', 'T1573.002', 'T1008', 'T1105', 'T1104', 'T1095', 'T1571',
         'T1572', 'T1090', 'T1090.001', 'T1090.002', 'T1090.003', 'T1090.004',
         'T1219', 'T1205', 'T1205.001', 'T1102', 'T1102.001', 'T1102.002',
         'T1102.003']),
    "TA0040": pd.Series(
        ['T1531', 'T1485', 'T1486', 'T1565', 'T1565.001', 'T1565.002', 'T1565.003',
         'T1491', 'T1491.001', 'T1491.002', 'T1561', 'T1561.001', 'T1561.002',
         'T1499', 'T1499.001', 'T1499.002', 'T1499.003', 'T1499.004', 'T1495',
         'T1490', 'T1498', 'T1498.001', 'T1498.002', 'T1496', 'T1489', 'T1529']),
    "TA0043": pd.Series(
        ['T1595', 'T1595.001', 'T1595.002', 'T1595.003', 'T1592', 'T1592.001',
         'T1592.002', 'T1592.003', 'T1592.004', 'T1589', 'T1589.001', 'T1589.002',
         'T1589.003', 'T1590', 'T1590.001', 'T1590.002', 'T1590.003', 'T1590.004',
         'T1590.005', 'T1590.006', 'T1591', 'T1591.001', 'T1591.002', 'T1591.003',
         'T1591.004', 'T1598', 'T1598.001', 'T1598.002', 'T1598.003', 'T1597',
         'T1597.001', 'T1597.002', 'T1596', 'T1596.001', 'T1596.002', 'T1596.003',
         'T1596.004', 'T1596.005', 'T1593', 'T1593.001', 'T1593.002', 'T1594']),
    "TA0042": pd.Series(
        ['T1583', 'T1583.001', 'T1583.002', 'T1583.003', 'T1583.004', 'T1583.005',
         'T1583.006', 'T1586', 'T1586.001', 'T1586.002', 'T1584', 'T1584.001',
         'T1584.002', 'T1584.003', 'T1584.004', 'T1584.005', 'T1584.006', 'T1587',
         'T1587.001', 'T1587.002', 'T1587.003', 'T1587.004', 'T1585', 'T1585.001',
         'T1585.002', 'T1588', 'T1588.001', 'T1588.002', 'T1588.003', 'T1588.004',
         'T1588.005', 'T1588.006', 'T1608', 'T1608.001', 'T1608.002', 'T1608.003',
         'T1608.004', 'T1608.005'])
}
