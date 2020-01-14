"""
-*- coding:utf-8 -*-
@Author  :   liaoyu
@Contact :   doitliao@126.com
@File    :   main.py
@Time    :   2020/1/11 01:44
@Desc    :
"""
import tushare as ts

token = "2e58fe8cff281e93cea76e048f47763d14fe3a23d262ad22c9d85b1c"

if __name__ == "__main__":
    pro = ts.pro_api(token)
    # ts.set_token(token)
    '''
    名称	    类型	 描述
    ts_code	str	股票代码
    trade_date	str	交易日期
    open	float	开盘价
    high	float	最高价
    low	float	最低价
    close	float	收盘价
    pre_close	float	昨收价
    change	float	涨跌额
    pct_chg	float	涨跌幅 （未复权，如果是复权请用 通用行情接口 ）
    vol	float	成交量 （手）
    amount	float	成交额 （千元）
    '''
    df = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20190718')
    print(df)

    data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,fullname, area,industry,list_date')
    data.to_parquet("stock_basic.parquet.gzip")
    print(data)