# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

from typing import Dict, List

from nuscenes import NuScenes

train_detect = \
    ['carla-1151', 'carla-1152', 'carla-1153', 'carla-1154', 'carla-1155', 'carla-1156', 'carla-1157', 'carla-1158',
     'carla-1159', 'carla-1160', 'carla-1161', 'carla-1162', 'carla-1163', 'carla-1164', 'carla-1165', 'carla-1166',
     'carla-1167', 'carla-1168', 'carla-1169', 'carla-1170', 'carla-1171', 'carla-1172', 'carla-1173', 'carla-1174',
     'carla-1175', 'carla-1176', 'carla-1177', 'carla-1178', 'carla-1179', 'carla-1180', 'carla-1181', 'carla-1182',
     'carla-1183', 'carla-1184', 'carla-1185', 'carla-1186', 'carla-1187', 'carla-1188', 'carla-1189', 'carla-1190',
     'carla-1191', 'carla-1192', 'carla-1193', 'carla-1194', 'carla-1195', 'carla-1196', 'carla-1197', 'carla-1198',
     'carla-1199', 'carla-1200', 'carla-1201', 'carla-1202', 'carla-1203', 'carla-1204', 'carla-1205', 'carla-1206',
     'carla-1207', 'carla-1208', 'carla-1209', 'carla-1210', 'carla-1211', 'carla-1212', 'carla-1213', 'carla-1214',
     'carla-1215', 'carla-1216', 'carla-1217', 'carla-1218', 'carla-1219', 'carla-1220', 'carla-1221', 'carla-1222',
     'carla-1223', 'carla-1224', 'carla-1225', 'carla-1226', 'carla-1227', 'carla-1228', 'carla-1229', 'carla-1230',
     'carla-1231', 'carla-1232', 'carla-1233', 'carla-1234', 'carla-1235', 'carla-1236', 'carla-1237', 'carla-1238',
     'carla-1239', 'carla-1240', 'carla-1241', 'carla-1242', 'carla-1243', 'carla-1244', 'carla-1245', 'carla-1246',
     'carla-1247', 'carla-1248', 'carla-1249', 'carla-1250', 'carla-1251', 'carla-1252', 'carla-1253', 'carla-1254',
     'carla-1255', 'carla-1256', 'carla-1257', 'carla-1258', 'carla-1259', 'carla-1260', 'carla-1261', 'carla-1262',
     'carla-1263', 'carla-1264', 'carla-1265', 'carla-1266', 'carla-1267', 'carla-1268', 'carla-1269', 'carla-1270',
     'carla-1271', 'carla-1272', 'carla-1273', 'carla-1274', 'carla-1275', 'carla-1276', 'carla-1277', 'carla-1278',
     'carla-1279', 'carla-1280', 'carla-1281', 'carla-1282', 'carla-1283', 'carla-1284', 'carla-1285', 'carla-1286',
     'carla-1287', 'carla-1288', 'carla-1289', 'carla-1290', 'carla-1291', 'carla-1292', 'carla-1293', 'carla-1294',
     'carla-1295', 'carla-1296', 'carla-1297', 'carla-1298', 'carla-1299', 'carla-1300', 'carla-1301', 'carla-1302',
     'carla-1303', 'carla-1304', 'carla-1305', 'carla-1306', 'carla-1307', 'carla-1308', 'carla-1309', 'carla-1310',
     'carla-1311', 'carla-1312', 'carla-1313', 'carla-1314', 'carla-1315', 'carla-1316', 'carla-1317', 'carla-1318',
     'carla-1319', 'carla-1320', 'carla-1321', 'carla-1322', 'carla-1323', 'carla-1324', 'carla-1325', 'carla-1326',
     'carla-1327', 'carla-1328', 'carla-1329', 'carla-1330', 'carla-1331', 'carla-1332', 'carla-1333', 'carla-1334',
     'carla-1335', 'carla-1336', 'carla-1337', 'carla-1338', 'carla-1339', 'carla-1340', 'carla-1341', 'carla-1342',
     'carla-1343', 'carla-1344', 'carla-1345', 'carla-1346', 'carla-1347', 'carla-1348', 'carla-1349', 'carla-1350',
     'carla-1351', 'carla-1352', 'carla-1353', 'carla-1354', 'carla-1355', 'carla-1356', 'carla-1357', 'carla-1358',
     'carla-1359', 'carla-1360', 'carla-1361', 'carla-1362', 'carla-1363', 'carla-1364', 'carla-1365', 'carla-1366',
     'carla-1367', 'carla-1368', 'carla-1369', 'carla-1370', 'carla-1371', 'carla-1372', 'carla-1373', 'carla-1374',
     'carla-1375', 'carla-1376', 'carla-1377', 'carla-1378', 'carla-1379', 'carla-1380', 'carla-1381', 'carla-1382',
     'carla-1383', 'carla-1384', 'carla-1385', 'carla-1386', 'carla-1387', 'carla-1388', 'carla-1389', 'carla-1390',
     'carla-1391', 'carla-1392', 'carla-1393', 'carla-1394', 'carla-1395', 'carla-1396', 'carla-1397', 'carla-1398',
     'carla-1399', 'carla-1400', 'carla-1401', 'carla-1402', 'carla-1403', 'carla-1404', 'carla-1405', 'carla-1406',
     'carla-1407', 'carla-1408', 'carla-1409', 'carla-1410', 'carla-1411', 'carla-1412', 'carla-1413', 'carla-1414',
     'carla-1415', 'carla-1416', 'carla-1417', 'carla-1418', 'carla-1419', 'carla-1420', 'carla-1421', 'carla-1422',
     'carla-1423', 'carla-1424', 'carla-1425', 'carla-1426', 'carla-1427', 'carla-1428', 'carla-1429', 'carla-1430',
     'carla-1431', 'carla-1432', 'carla-1433', 'carla-1434', 'carla-1435', 'carla-1436', 'carla-1437', 'carla-1438',
     'carla-1439', 'carla-1440', 'carla-1441', 'carla-1442', 'carla-1443', 'carla-1444', 'carla-1445', 'carla-1446',
     'carla-1447', 'carla-1448', 'carla-1449', 'carla-1450', 'carla-1451', 'carla-1452', 'carla-1453', 'carla-1454',
     'carla-1455', 'carla-1456', 'carla-1457', 'carla-1458', 'carla-1459', 'carla-1460', 'carla-1461', 'carla-1462',
     'carla-1463', 'carla-1464', 'carla-1465', 'carla-1466', 'carla-1467', 'carla-1468', 'carla-1469', 'carla-1470',
     'carla-1471', 'carla-1472', 'carla-1473', 'carla-1474', 'carla-1475', 'carla-1476', 'carla-1477', 'carla-1478',
     'carla-1479', 'carla-1480', 'carla-1481', 'carla-1482', 'carla-1483', 'carla-1484', 'carla-1485', 'carla-1486',
     'carla-1487', 'carla-1488', 'carla-1489', 'carla-1490', 'carla-1491', 'carla-1492', 'carla-1493', 'carla-1494',
     'carla-1495', 'carla-1496', 'carla-1497', 'carla-1498', 'carla-1499', 'carla-1500', 'carla-1501', 'carla-1502',
     'carla-1503', 'carla-1504', 'carla-1505', 'carla-1506', 'carla-1507', 'carla-1508', 'carla-1509', 'carla-1510',
     'carla-1511', 'carla-1512', 'carla-1513', 'carla-1514', 'carla-1515', 'carla-1516', 'carla-1517', 'carla-1518',
     'carla-1519', 'carla-1520', 'carla-1521', 'carla-1522', 'carla-1523', 'carla-1524', 'carla-1525', 'carla-1526',
     'carla-1527', 'carla-1528', 'carla-1529', 'carla-1530', 'carla-1531', 'carla-1532', 'carla-1533', 'carla-1534',
     'carla-1535', 'carla-1536', 'carla-1537', 'carla-1538', 'carla-1539', 'carla-1540', 'carla-1541', 'carla-1542',
     'carla-1543', 'carla-1544', 'carla-1545', 'carla-1546', 'carla-1547', 'carla-1548', 'carla-1549', 'carla-1550',
     'carla-1551', 'carla-1552', 'carla-1553', 'carla-1554', 'carla-1555', 'carla-1556', 'carla-1557', 'carla-1558',
     'carla-1559', 'carla-1560', 'carla-1561', 'carla-1562', 'carla-1563', 'carla-1564', 'carla-1565', 'carla-1566',
     'carla-1567', 'carla-1568', 'carla-1569', 'carla-1570', 'carla-1571', 'carla-1572', 'carla-1573', 'carla-1574',
     'carla-1575', 'carla-1576', 'carla-1577', 'carla-1578', 'carla-1579', 'carla-1580', 'carla-1581', 'carla-1582',
     'carla-1583', 'carla-1584', 'carla-1585', 'carla-1586', 'carla-1587', 'carla-1588', 'carla-1589', 'carla-1590',
     'carla-1591', 'carla-1592', 'carla-1593', 'carla-1594', 'carla-1595', 'carla-1596', 'carla-1597', 'carla-1598',
     'carla-1599', 'carla-1600', 'carla-1601', 'carla-1602', 'carla-1603', 'carla-1604', 'carla-1605', 'carla-1606',
     'carla-1607', 'carla-1608', 'carla-1609', 'carla-1610', 'carla-1611', 'carla-1612', 'carla-1613', 'carla-1614',
     'carla-1615', 'carla-1616', 'carla-1617', 'carla-1618', 'carla-1619', 'carla-1620', 'carla-1621', 'carla-1622',
     'carla-1623', 'carla-1624', 'carla-1625', 'carla-1626', 'carla-1627', 'carla-1628', 'carla-1629', 'carla-1630',
     'carla-1631', 'carla-1632', 'carla-1633', 'carla-1634', 'carla-1635', 'carla-1636', 'carla-1637', 'carla-1638',
     'carla-1639', 'carla-1640', 'carla-1641', 'carla-1642', 'carla-1643', 'carla-1644', 'carla-1645', 'carla-1646',
     'carla-1647', 'carla-1648', 'carla-1649', 'carla-1650', 'carla-1651', 'carla-1652', 'carla-1653', 'carla-1654',
     'carla-1655', 'carla-1656', 'carla-1657', 'carla-1658', 'carla-1659', 'carla-1660', 'carla-1661', 'carla-1662',
     'carla-1663', 'carla-1664', 'carla-1665', 'carla-1666', 'carla-1667', 'carla-1668', 'carla-1669', 'carla-1670',
     'carla-1671', 'carla-1672', 'carla-1673', 'carla-1674', 'carla-1675', 'carla-1676', 'carla-1677', 'carla-1678',
     'carla-1679', 'carla-1680', 'carla-1681', 'carla-1682', 'carla-1683', 'carla-1684', 'carla-1685', 'carla-1686',
     'carla-1687', 'carla-1688', 'carla-1689', 'carla-1690', 'carla-1691', 'carla-1692', 'carla-1693', 'carla-1694',
     'carla-1695', 'carla-1696', 'carla-1697', 'carla-1698', 'carla-1699', 'carla-1700', 'carla-1701', 'carla-1702',
     'carla-1703', 'carla-1704', 'carla-1705', 'carla-1706', 'carla-1707', 'carla-1708', 'carla-1709', 'carla-1710',
     'carla-1711', 'carla-1712', 'carla-1713', 'carla-1714', 'carla-1715', 'carla-1716', 'carla-1717', 'carla-1718',
     'carla-1719', 'carla-1720', 'carla-1721', 'carla-1722', 'carla-1723', 'carla-1724', 'carla-1725', 'carla-1726',
     'carla-1727', 'carla-1728', 'carla-1729', 'carla-1730', 'carla-1731', 'carla-1732', 'carla-1733', 'carla-1734',
     'carla-1735', 'carla-1736', 'carla-1737', 'carla-1738', 'carla-1739', 'carla-1740', 'carla-1741', 'carla-1742',
     'carla-1743', 'carla-1744', 'carla-1745', 'carla-1746', 'carla-1747', 'carla-1748', 'carla-1749', 'carla-1750',
     'carla-1751', 'carla-1752', 'carla-1753', 'carla-1754', 'carla-1755', 'carla-1756', 'carla-1757', 'carla-1758',
     'carla-1759', 'carla-1760', 'carla-1761', 'carla-1762', 'carla-1763', 'carla-1764', 'carla-1765', 'carla-1766',
     'carla-1767', 'carla-1768', 'carla-1769', 'carla-1770', 'carla-1771', 'carla-1772', 'carla-1773', 'carla-1774',
     'carla-1775', 'carla-1776', 'carla-1777', 'carla-1778', 'carla-1779', 'carla-1780', 'carla-1781', 'carla-1782',
     'carla-1783', 'carla-1784', 'carla-1785', 'carla-1786', 'carla-1787', 'carla-1788', 'carla-1789', 'carla-1790',
     'carla-1791', 'carla-1792', 'carla-1793', 'carla-1794', 'carla-1795', 'carla-1796', 'carla-1797', 'carla-1798',
     'carla-1799', 'carla-1800', 'carla-1801', 'carla-1802', 'carla-1803', 'carla-1804', 'carla-1805', 'carla-1806',
     'carla-1807', 'carla-1808', 'carla-1809', 'carla-1810', 'carla-1811', 'carla-1812', 'carla-1813', 'carla-1814',
     'carla-1815', 'carla-1816', 'carla-1817', 'carla-1818', 'carla-1819', 'carla-1820', 'carla-1821', 'carla-1822',
     'carla-1823', 'carla-1824', 'carla-1825', 'carla-1826', 'carla-1827', 'carla-1828', 'carla-1829', 'carla-1830',
     'carla-1831', 'carla-1832', 'carla-1833', 'carla-1834', 'carla-1835', 'carla-1836', 'carla-1837', 'carla-1838',
     'carla-1839', 'carla-1840', 'carla-1841', 'carla-1842', 'carla-1843', 'carla-1844', 'carla-1845', 'carla-1846',
     'carla-1847', 'carla-1848', 'carla-1849', 'carla-1850']

train_track = \
    []

train = list(sorted(set(train_detect + train_track)))

val = \
    ['carla-1001', 'carla-1002', 'carla-1003', 'carla-1004', 'carla-1005', 'carla-1006', 'carla-1007', 'carla-1008',
     'carla-1009', 'carla-1010', 'carla-1011', 'carla-1012', 'carla-1013', 'carla-1014', 'carla-1015', 'carla-1016',
     'carla-1017', 'carla-1018', 'carla-1019', 'carla-1020', 'carla-1021', 'carla-1022', 'carla-1023', 'carla-1024',
     'carla-1025', 'carla-1026', 'carla-1027', 'carla-1028', 'carla-1029', 'carla-1030', 'carla-1031', 'carla-1032',
     'carla-1033', 'carla-1034', 'carla-1035', 'carla-1036', 'carla-1037', 'carla-1038', 'carla-1039', 'carla-1040',
     'carla-1041', 'carla-1042', 'carla-1043', 'carla-1044', 'carla-1045', 'carla-1046', 'carla-1047', 'carla-1048',
     'carla-1049', 'carla-1050', 'carla-1051', 'carla-1052', 'carla-1053', 'carla-1054', 'carla-1055', 'carla-1056',
     'carla-1057', 'carla-1058', 'carla-1059', 'carla-1060', 'carla-1061', 'carla-1062', 'carla-1063', 'carla-1064',
     'carla-1065', 'carla-1066', 'carla-1067', 'carla-1068', 'carla-1069', 'carla-1070', 'carla-1071', 'carla-1072',
     'carla-1073', 'carla-1074', 'carla-1075', 'carla-1076', 'carla-1077', 'carla-1078', 'carla-1079', 'carla-1080',
     'carla-1081', 'carla-1082', 'carla-1083', 'carla-1084', 'carla-1085', 'carla-1086', 'carla-1087', 'carla-1088',
     'carla-1089', 'carla-1090', 'carla-1091', 'carla-1092', 'carla-1093', 'carla-1094', 'carla-1095', 'carla-1096',
     'carla-1097', 'carla-1098', 'carla-1099', 'carla-1100', 'carla-1101', 'carla-1102', 'carla-1103', 'carla-1104',
     'carla-1105', 'carla-1106', 'carla-1107', 'carla-1108', 'carla-1109', 'carla-1110', 'carla-1111', 'carla-1112',
     'carla-1113', 'carla-1114', 'carla-1115', 'carla-1116', 'carla-1117', 'carla-1118', 'carla-1119', 'carla-1120',
     'carla-1121', 'carla-1122', 'carla-1123', 'carla-1124', 'carla-1125', 'carla-1126', 'carla-1127', 'carla-1128',
     'carla-1129', 'carla-1130', 'carla-1131', 'carla-1132', 'carla-1133', 'carla-1134', 'carla-1135', 'carla-1136',
     'carla-1137', 'carla-1138', 'carla-1139', 'carla-1140', 'carla-1141', 'carla-1142', 'carla-1143', 'carla-1144',
     'carla-1145', 'carla-1146', 'carla-1147', 'carla-1148', 'carla-1149', 'carla-1150']

test = \
    []

mini_train = \
    []

mini_val = \
    []


def create_splits_logs(split: str, nusc: 'NuScenes') -> List[str]:
    """
    Returns the logs in each dataset split of nuScenes.
    Note: Previously this script included the teaser dataset splits. Since new scenes from those logs were added and
          others removed in the full dataset, that code is incompatible and was removed.
    :param split: NuScenes split.
    :param nusc: NuScenes instance.
    :return: A list of logs in that split.
    """
    # Load splits on a scene-level.
    scene_splits = create_splits_scenes(verbose=False)

    assert split in scene_splits.keys(), 'Requested split {} which is not a known nuScenes split.'.format(split)

    # Check compatibility of split with nusc_version.
    version = nusc.version
    if split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    elif split == 'test':
        assert version.endswith('test'), \
            'Requested split {} which is not compatible with NuScenes version {}'.format(split, version)
    else:
        raise ValueError('Requested split {} which this function cannot map to logs.'.format(split))

    # Get logs for this split.
    scene_to_log = {scene['name']: nusc.get('log', scene['log_token'])['logfile'] for scene in nusc.scene}
    logs = set()
    scenes = scene_splits[split]
    for scene in scenes:
        logs.add(scene_to_log[scene])

    return list(logs)


def create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:
    """
    Similar to create_splits_logs, but returns a mapping from split to scene names, rather than log names.
    The splits are as follows:
    - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
    - train_detect/train_track: Two halves of the train split used for separating the training sets of detector and
        tracker if required.
    :param verbose: Whether to print out statistics on a scene level.
    :return: A mapping from split name to a list of scenes names in that split.
    """
    # Use hard-coded splits.
    all_scenes = train + val + test
    # assert len(all_scenes) == 1000 and len(set(all_scenes)) == 1000, 'Error: Splits incomplete!'
    scene_splits = {'train': train, 'val': val, 'test': test,
                    'mini_train': mini_train, 'mini_val': mini_val,
                    'train_detect': train_detect, 'train_track': train_track}

    # Optional: Print scene-level stats.
    if verbose:
        for split, scenes in scene_splits.items():
            print('%s: %d' % (split, len(scenes)))
            print('%s' % scenes)

    return scene_splits


if __name__ == '__main__':
    # Print the scene-level stats.
    create_splits_scenes(verbose=True)
